"""Measure a league's final-margin standard deviation from played games.

MARGIN_STD_DEV turns a win probability into a points line, and every entry in
it was measured from graded results except NBA and NFL, which "keep published
figures" because neither had graded games in this repo yet. That exception is
load-bearing and worth removing: NBA's 11.5 sits below the measured WNBA 13.49
for a higher-scoring version of the same sport, which is the wrong direction.

The likely explanation is that 11.5 is the standard deviation of margin around
the closing spread -- a residual -- rather than of the raw margin, which is what
the map is documented to hold and what the probability-to-line conversion needs.
This script reports both, so the two are never confused again.

Seasons that have already been played are available from ESPN's scoreboard, so
there is no need to wait for tip-off to measure this. Run it from the `ESPN
reachability probe` workflow -- the dev sandbox's egress policy blocks
espn.com, so it cannot be run locally.

    python scripts/measure_margin_sd.py nba 2025-11-01 2026-03-31
"""

from __future__ import annotations

import datetime
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sports_config import get_league  # noqa: E402

USER_AGENT = "EdgeBoard/1.0 (+https://github.com/bradleyschulz88/Predictions-Model)"
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={date}"


def _iter_dates(start: str, end: str):
    day = datetime.date.fromisoformat(start)
    last = datetime.date.fromisoformat(end)
    while day <= last:
        yield day
        day += datetime.timedelta(days=1)


def _final_margins(payload: dict) -> list[tuple[float, float | None]]:
    """Return (home margin, home spread) for each completed game on a slate."""
    out: list[tuple[float, float | None]] = []
    for event in payload.get("events") or []:
        for competition in event.get("competitions") or []:
            status = ((competition.get("status") or {}).get("type") or {})
            if not status.get("completed"):
                continue
            home = away = None
            for competitor in competition.get("competitors") or []:
                score = competitor.get("score")
                try:
                    value = float(score)
                except (TypeError, ValueError):
                    continue
                if competitor.get("homeAway") == "home":
                    home = value
                elif competitor.get("homeAway") == "away":
                    away = value
            if home is None or away is None:
                continue

            # ESPN quotes the spread from the favourite's side, so read the
            # home team's own handicap rather than the headline number.
            spread = None
            for odds in competition.get("odds") or []:
                home_odds = odds.get("homeTeamOdds") or {}
                for key in ("spreadOdds", "pointSpread", "spread"):
                    raw = home_odds.get(key)
                    if isinstance(raw, dict):
                        raw = raw.get("value") or raw.get("displayValue")
                    try:
                        spread = float(raw)
                        break
                    except (TypeError, ValueError):
                        continue
                if spread is None:
                    try:
                        spread = float(odds.get("spread"))
                    except (TypeError, ValueError):
                        spread = None
                if spread is not None:
                    break
            out.append((home - away, spread))
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    league, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    path = get_league(league).espn_path

    margins: list[float] = []
    residuals: list[float] = []
    days = failures = 0
    for day in _iter_dates(start, end):
        url = SCOREBOARD.format(path=path, date=day.strftime("%Y%m%d"))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError):
            failures += 1
            continue
        days += 1
        for margin, spread in _final_margins(payload):
            margins.append(margin)
            if spread is not None:
                # spread is the home handicap, so margin + spread is the
                # amount by which the home side beat its own line.
                residuals.append(margin + spread)

    print(f"{league}: {start} to {end}")
    print(f"  slates fetched {days}, failed {failures}")
    if len(margins) < 2:
        print("  not enough completed games to measure")
        return 1

    print(f"  raw final margin   n={len(margins):5}  "
          f"mean {statistics.fmean(margins):+6.2f}  SD {statistics.stdev(margins):6.2f}")
    if len(residuals) >= 2:
        print(f"  margin vs spread   n={len(residuals):5}  "
              f"mean {statistics.fmean(residuals):+6.2f}  SD {statistics.stdev(residuals):6.2f}")
    else:
        print("  margin vs spread   no spreads on these slates")
    print()
    print("  MARGIN_STD_DEV holds the raw final margin SD, the first line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
