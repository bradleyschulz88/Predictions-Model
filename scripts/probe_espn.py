"""Ask ESPN, from a runner, which request shapes it currently accepts.

Written during the 2026-08-04 outage, when Akamai began returning 403 to every
site.api.espn.com request and took all six leagues down at once. The build kept
succeeding and publishing empty slates, so the only visible symptom was on the
site. Diagnosis took three guesses; this script would have taken one.

Kept because that failure mode will recur: bot rules change without notice, the
dev sandbox's egress policy blocks espn.com so it cannot be tested locally, and
a runner is the only place the real answer lives. Each header profile is tried
against each host, then repeated so a probabilistic rule cannot masquerade as a
clean pass.

Run via the `ESPN reachability probe` workflow.
"""

from __future__ import annotations

import datetime
import json
import ssl
import urllib.error
import urllib.request

# Today in US Eastern, which is the day ESPN's scoreboard keys on. A fixed
# date would quietly start probing an empty slate and read as a change in
# ESPN's behaviour when it is only a stale constant.
TODAY = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=5)).strftime("%Y%m%d")

HOSTS = {
    "site.api scoreboard (what the build uses)": (
        f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={TODAY}"
    ),
    "site.api teams (enrichment path)": (
        "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/teams"
    ),
    "sports.core.api (odds path)": (
        "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events"
    ),
    "cdn.espn.com (alternate scoreboard)": (
        f"https://cdn.espn.com/core/mlb/scoreboard?xhr=1&date={TODAY}"
    ),
}

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

PROFILES: dict[str, dict[str, str]] = {
    # Exactly what sbr_client.get_text sends today.
    "production UA": {"User-Agent": "Mozilla/5.0 (compatible; MLB-SBR-Client/1.0)"},
    # What data_providers/utils.fetch_json already sends on other calls.
    "browser UA": {"User-Agent": BROWSER_UA},
    "browser UA + accept headers": {
        "User-Agent": BROWSER_UA,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.espn.com/",
        "Origin": "https://www.espn.com",
    },
    # The first probe showed Akamai rejects every UA that claims to be Mozilla
    # while allowing urllib's honest default, so the rule is aimed at spoofed
    # browsers rather than at scripts. These candidates check how far that
    # tolerance extends, because the replacement UA has to be one that is
    # tested rather than assumed.
    "urllib default UA": {},
    "explicit Python-urllib": {"User-Agent": "Python-urllib/3.12"},
    "honest project UA": {
        "User-Agent": "EdgeBoard/1.0 (+https://github.com/bradleyschulz88/Predictions-Model)"
    },
    "honest project UA, bare": {"User-Agent": "EdgeBoard/1.0"},
}


def probe(url: str, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers)
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            body = response.read()
            return f"{response.status} OK ({len(body):,} bytes)"
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            snippet = exc.read()[:160].decode("utf-8", "replace").replace("\n", " ")
            if snippet.strip():
                detail = f" | body: {snippet.strip()}"
        except Exception:  # noqa: BLE001 - diagnostics must not mask the status
            pass
        server = exc.headers.get("Server") or "?"
        reference = exc.headers.get("X-Reference-Error") or exc.headers.get("Cf-Ray") or ""
        marker = f" | server={server}"
        if reference:
            marker += f" ref={reference}"
        return f"HTTP {exc.code} {exc.reason}{marker}{detail}"
    except urllib.error.URLError as exc:
        return f"network error: {exc.reason}"


def egress_ip() -> str:
    try:
        with urllib.request.urlopen("https://api.ipify.org?format=json", timeout=15) as response:
            return json.loads(response.read()).get("ip", "?")
    except Exception as exc:  # noqa: BLE001 - best effort only
        return f"unavailable ({exc})"


REPEATS = 5


def main() -> None:
    print(f"Runner egress IP: {egress_ip()}")
    print()
    for host_label, url in HOSTS.items():
        print(host_label)
        print(f"  {url}")
        for profile_label, headers in PROFILES.items():
            print(f"    {profile_label:<32} {probe(url, headers)}")
        print()

    # A single request per profile is not enough to choose the replacement UA:
    # if Akamai's rule is probabilistic, one 200 proves nothing. Repeat the
    # candidates against the host that is actually blocked and report the
    # tally, so the UA that ships is the one that passed every time.
    blocked_url = HOSTS["site.api scoreboard (what the build uses)"]
    print(f"Repeatability on the blocked host, {REPEATS} requests each")
    for profile_label, headers in PROFILES.items():
        outcomes = [probe(blocked_url, headers) for _ in range(REPEATS)]
        ok = sum(1 for outcome in outcomes if outcome.startswith("200"))
        codes = sorted({outcome.split(" |")[0] for outcome in outcomes})
        print(f"    {profile_label:<32} {ok}/{REPEATS} ok  ({', '.join(codes)})")


if __name__ == "__main__":
    main()
