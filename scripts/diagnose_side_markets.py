"""What does ESPN core actually return for one event's odds?

Written because re-enabling the side-market pass fixed totals and did not fix
runlines. On the 2026-08-05 22:46Z build the pass priced 85 of 86 totals but
0 of 62 MLB runlines, while WNBA and AFL spreads priced fine. Those MLB events
were fetched successfully -- their totals came from the same response -- so
ESPN core is returning something for the spread that this code does not use.

Three candidates, and guessing between them is how the last two diagnoses went
wrong:

  1. ESPN core returns no spread row at all for baseball.
  2. It returns one under a viewType that does not contain "Spread", so the
     filter in fill_missing_side_market_prices drops it.
  3. It returns one that survives the filter but whose price
     extract_spread_price cannot parse.

This prints the raw viewTypes and currentLine shapes for one event per league,
then runs the real extractors over them, so the answer is read rather than
assumed. The dev sandbox's egress policy blocks espn.com, so run it from the
`ESPN reachability check` workflow.

    python scripts/diagnose_side_markets.py mlb 2026-08-06
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from espn_client import fetch_scoreboard, parse_scoreboard  # noqa: E402
from espn_odds import fetch_event_odds, has_priced_market  # noqa: E402
from mlb_predictions import extract_spread_price, extract_total_price  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    league, date_value = sys.argv[1], sys.argv[2]

    games = parse_scoreboard(fetch_scoreboard(league, date_value), league=league)
    if not games:
        print(f"{league} {date_value}: no games on the scoreboard")
        return 1
    print(f"{league} {date_value}: {len(games)} games")

    game = games[0]
    event_id = game.get("eventId")
    print(f"sampling event {event_id}: {game.get('awayTeam')} @ {game.get('homeTeam')}")
    print()

    print("what the scoreboard/SBR path already put on the game")
    for line in game.get("lines") or []:
        print(f"    {line.get('sportsbook'):<18} {line.get('viewType'):<12} "
              f"{json.dumps(line.get('currentLine'))}")
    for view in ("Total", "Spread"):
        print(f"    has a priced {view}: {has_priced_market(game.get('lines') or [], view)}")
    print()

    fetched = fetch_event_odds(league, event_id)
    print(f"what ESPN core returns for that event: {len(fetched)} rows")
    for line in fetched:
        print(f"    {str(line.get('sportsbook')):<18} {str(line.get('viewType')):<12} "
              f"{json.dumps(line.get('currentLine'))}")
    print()

    # Candidate 2: does the filter keep it? This is the exact expression
    # fill_missing_side_market_prices uses.
    for view in ("Total", "Spread"):
        kept = [line for line in fetched if view in (line.get("viewType") or "")]
        print(f"    rows whose viewType contains {view!r}: {len(kept)}")

    # Candidate 3: can the extractors read a price out of what survived?
    print()
    print("what the extractors make of the merged lines")
    merged = (game.get("lines") or []) + fetched
    for side in ("over", "under"):
        print(f"    extract_total_price({side!r})  = {extract_total_price(merged, side)}")
    for side in ("home", "away"):
        print(f"    extract_spread_price({side!r}) = {extract_spread_price(merged, side)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
