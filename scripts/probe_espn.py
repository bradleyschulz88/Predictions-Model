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

Two audiences, in that order. The table is for a human choosing a replacement
User-Agent. The exit code is for the scheduler: this returns non-zero when the
UA the build actually sends stops getting through, which is the alarm that did
not exist on 2026-08-04 -- the build kept succeeding and publishing empty
slates for half a day, because nothing about an empty slate is an error.

Run via the `ESPN reachability check` workflow.
"""

from __future__ import annotations

import datetime
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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


def probe_predictor_coverage() -> None:
    """Report whether live summaries still carry the Matchup Predictor.

    Every build warns that ESPN predictor coverage is 0% for MLB, and it did so
    before the 403 outage too, so it is a separate fault. The parser handles the
    shape in tests/fixtures/espn_summary_401815776.json correctly, which leaves
    two candidates: summaries are not being fetched, or live payloads no longer
    carry the field. Only ESPN can settle that.
    """
    headers = {"User-Agent": "EdgeBoard/1.0 (+https://github.com/bradleyschulz88/Predictions-Model)"}
    scoreboard_url = (
        f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={TODAY}"
    )
    print("Live summary shape, MLB")
    try:
        request = urllib.request.Request(scoreboard_url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            events = json.loads(response.read()).get("events") or []
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"    could not list events: {exc}")
        return
    if not events:
        print("    no events on this slate, nothing to sample")
        return

    for event in events[:3]:
        event_id = event.get("id")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary?event={event_id}"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                summary = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            print(f"    event {event_id}: {exc}")
            continue
        predictor = summary.get("predictor") or {}
        home = predictor.get("homeTeam") or {}
        print(
            f"    event {event_id}: predictor={'yes' if predictor else 'NO'} "
            f"homeTeam keys={sorted(home) if home else '-'} "
            f"pickcenter={len(summary.get('pickcenter') or [])} "
            f"winprobability={len(summary.get('winprobability') or [])}"
        )
    print()


def check_shipping_user_agent(url: str) -> bool:
    """Does the User-Agent the build actually sends still get through?

    The report above is for a human choosing a fix. This is the part a machine
    can act on, and it is why the workflow is worth keeping on a schedule
    rather than deleting now the 2026-08-04 cause is known: bot rules change
    without notice, and the last time one did, the build kept succeeding and
    publishing empty slates for half a day. Nothing failed, so nothing said
    anything. A red run here is the warning that did not exist.

    Repeated, because a probabilistic rule that lets one request through would
    otherwise read as healthy.
    """
    from espn_client import ESPN_USER_AGENT

    outcomes = [probe(url, {"User-Agent": ESPN_USER_AGENT}) for _ in range(REPEATS)]
    ok = sum(1 for outcome in outcomes if outcome.startswith("200"))
    print()
    print(f"Shipping User-Agent ({ESPN_USER_AGENT}): {ok}/{REPEATS} ok")
    if ok == REPEATS:
        return True
    for outcome in outcomes:
        if not outcome.startswith("200"):
            print(f"  {outcome}")
    return False


def main() -> int:
    print(f"Runner egress IP: {egress_ip()}")
    print()
    probe_predictor_coverage()
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

    if check_shipping_user_agent(blocked_url):
        return 0
    print()
    print("::error::ESPN is rejecting the User-Agent the build sends. The board "
          "will publish empty slates without failing. See the table above for a "
          "profile that still works and update ESPN_USER_AGENT in espn_client.py.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
