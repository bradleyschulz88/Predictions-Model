"""Probe ESPN from a GitHub runner to find out why every request returns 403.

The scheduled build started failing with HTTP 403 on every scoreboard request
between 2026-08-04 11:05Z and 23:31Z. Two explanations fit: ESPN began
rejecting our bot-shaped User-Agent, or it began rejecting the runner's IP
range. Those need opposite fixes, so this asks ESPN directly.

Each header profile is tried against each host. A profile that succeeds where
the production one fails means headers are the problem and the fix is local. If
every profile fails identically, the request is not what ESPN objects to.

Run via the `ESPN reachability probe` workflow. Delete once the cause is known.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

TODAY = "20260805"

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
    # urllib's own default, to see whether any UA at all is the trigger.
    "urllib default UA": {},
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


def main() -> None:
    print(f"Runner egress IP: {egress_ip()}")
    print()
    for host_label, url in HOSTS.items():
        print(host_label)
        print(f"  {url}")
        for profile_label, headers in PROFILES.items():
            print(f"    {profile_label:<32} {probe(url, headers)}")
        print()


if __name__ == "__main__":
    main()
