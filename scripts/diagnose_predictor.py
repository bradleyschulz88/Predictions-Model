"""Find where the ESPN Matchup Predictor is lost between ESPN and coverage.

Every build warns that predictor coverage is 0%, and it did so before the 403
outage, so it is not a casualty of that. Static reading exonerated each stage
on its own: live summaries carry predictor.homeTeam.gameProjection, the parser
handles that shape against the checked-in fixture, enrichment runs before
predictions, and the cache is in-memory with a 300s TTL so nothing stale
survives a build. The value is still missing, so this walks the real production
path on a live game and prints the value at each hop.

Run from the `ESPN reachability probe` workflow -- the dev sandbox's egress
policy blocks espn.com, so it cannot be run locally.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_coverage import coverage_from_game, summarize_coverage  # noqa: E402
from espn_client import ESPN_USER_AGENT, build_scoreboard_url, parse_scoreboard  # noqa: E402
from espn_enrichment import enrich_game, fetch_event_summary  # noqa: E402
from mlb_predictions import apply_predictions  # noqa: E402


def main() -> int:
    league = sys.argv[1] if len(sys.argv) > 1 else "mlb"
    date = sys.argv[2] if len(sys.argv) > 2 else None

    url = build_scoreboard_url(league, date)
    request = urllib.request.Request(url, headers={"User-Agent": ESPN_USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        scoreboard = json.loads(response.read())

    games = parse_scoreboard(scoreboard, league=league)
    print(f"{league} {date or 'today'}: {len(games)} games from the scoreboard")
    if not games:
        return 1

    game = games[0]
    event_id = game.get("eventId")
    print(f"sampling event {event_id}: {game.get('awayTeam')} @ {game.get('homeTeam')}")
    print()

    # 1. Does the summary ESPN returns carry the field at all?
    summary = fetch_event_summary(event_id, league=league)
    predictor = summary.get("predictor") or {}
    print(f"1. summary.predictor present: {bool(predictor)}")
    if predictor:
        print(f"   homeTeam: {predictor.get('homeTeam')}")

    # 2. Does the production enrichment step extract it?
    enrich_game(game)
    enrichment = game.get("enrichment") or {}
    print(f"2. enrichment espnPredictorHome: {enrichment.get('espnPredictorHome')!r}")
    print(f"   enrichment espnPredictorAway: {enrichment.get('espnPredictorAway')!r}")
    print(f"   enrichment keys: {len(enrichment)}")

    # 3. Does it survive into the prediction's feature block?
    apply_predictions([game])
    features = ((game.get("prediction") or {}).get("features") or {})
    data_coverage = features.get("dataCoverage") or {}
    print(f"3. features.dataCoverage.espnPredictor: {data_coverage.get('espnPredictor')!r}")
    print(f"   dataCoverage present: {bool(data_coverage)}")

    # 4. What does the coverage summary conclude?
    print(f"4. coverage_from_game espnPredictor: {coverage_from_game(game).get('espnPredictor')!r}")
    print(f"   summarize_coverage pct: {summarize_coverage([game])['pct']['espnPredictor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
