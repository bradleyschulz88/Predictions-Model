"""Rebuild pre-game features for completed seasons, to screen candidates on more games.

The problem this solves
-----------------------
The fit has ~700 graded games. At roughly 10-20 games per predictor that
supports about five features, and there are fourteen candidates queued. Most
will never be judged on the live log alone -- a candidate added today needs
months before it has enough coverage to beat or lose to its own absence.

There are thousands of completed games sitting in ESPN's schedule. Replaying
them gives the ablation a far larger sample to screen against.

Leakage is the whole problem
----------------------------
A backfill that uses end-of-season records looks superb in backtest and fails
live, because the features encode the answer. Every number here is therefore
computed by **replaying the season in date order** and reading the state as it
stood *before* the game:

* records are accumulated from prior results only, never fetched;
* form is the previous five results, not including this one;
* rest is measured from the club's previous fixture;
* park and travel are static properties of who is playing where.

The final score is used for one thing only: the label. `_assert_no_leakage` and
its tests exist to keep it that way.

What it deliberately cannot do
------------------------------
**No odds, so no `marketLogit`.** Historical closing lines are not available
from any source this project uses, and inventing them would be worse than
omitting them. That makes this a screening set for the *standalone* model, not
a replacement for the live fit -- a candidate that cannot beat `strengthDiff`
alone on thousands of games is not going to earn a place alongside the market
anchor on seven hundred.

Usage
-----
    python scripts/backfill_history.py --league mlb --start 2025-04-01 --end 2025-09-28

Writes `docs/data/history_features.json`, which is gitignored and rebuilt on
demand -- it is derived data, not a record.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.park_factors import park_run_environment  # noqa: E402
from data_providers.travel import travel_edge  # noqa: E402
from espn_client import fetch_scoreboard, parse_scoreboard  # noqa: E402

HISTORY_FILE = "history_features.json"

# A club needs some games behind it before its record means anything. Below
# this the features are noise and the row is skipped rather than fitted on.
MIN_GAMES_BEFORE_USABLE = 10

FORM_WINDOW = 5


class TeamState:
    """Everything known about a club from its completed games so far."""

    __slots__ = ("wins", "losses", "home_wins", "home_losses", "away_wins",
                 "away_losses", "form", "last_played")

    def __init__(self) -> None:
        self.wins = 0
        self.losses = 0
        self.home_wins = 0
        self.home_losses = 0
        self.away_wins = 0
        self.away_losses = 0
        self.form: deque[int] = deque(maxlen=FORM_WINDOW)
        self.last_played: str | None = None

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def win_pct(self) -> float | None:
        return self.wins / self.games if self.games else None

    def split_pct(self, *, at_home: bool) -> float | None:
        """Home win rate for the home side, road win rate for the visitors."""
        wins = self.home_wins if at_home else self.away_wins
        losses = self.home_losses if at_home else self.away_losses
        total = wins + losses
        return wins / total if total else None

    def form_pct(self) -> float | None:
        return sum(self.form) / len(self.form) if self.form else None

    def record(self, won: bool, *, at_home: bool, played_on: str) -> None:
        if won:
            self.wins += 1
            if at_home:
                self.home_wins += 1
            else:
                self.away_wins += 1
        else:
            self.losses += 1
            if at_home:
                self.home_losses += 1
            else:
                self.away_losses += 1
        self.form.append(1 if won else 0)
        self.last_played = played_on


def _rest_days(state: TeamState, game_date: str) -> int | None:
    if not state.last_played:
        return None
    try:
        return (date.fromisoformat(game_date) - date.fromisoformat(state.last_played)).days
    except ValueError:
        return None


def _final_scores(game: dict[str, Any]) -> tuple[int, int] | None:
    """Home and away runs for a completed game, or None if it is not usable."""
    if not game.get("isFinal"):
        return None
    if game.get("isVoided") or game.get("isPostponed") or game.get("isCanceled"):
        return None
    try:
        return int(game["homeScore"]), int(game["awayScore"])
    except (KeyError, TypeError, ValueError):
        return None


def _assert_no_leakage(features: dict[str, Any], home_score: int, away_score: int) -> None:
    """Guard against the failure mode that makes a backfill worthless.

    A feature must not be derivable from this game's result. The cheapest useful
    check is that no feature equals the margin or either score, which catches
    the realistic mistakes -- writing the label in by accident, or recording the
    game into team state before reading the features back out.
    """
    forbidden = {float(home_score), float(away_score), float(home_score - away_score)}
    # Zero and small integers occur legitimately all over the feature set.
    forbidden = {value for value in forbidden if abs(value) > 3}
    for name, value in features.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value) in forbidden:
                raise AssertionError(
                    f"feature {name}={value} matches this game's score "
                    f"({home_score}-{away_score}); state was updated before "
                    "the features were read"
                )


def build_features(
    game: dict[str, Any],
    home_state: TeamState,
    away_state: TeamState,
    *,
    league: str,
) -> dict[str, Any] | None:
    """Pre-game features from state that predates the game. None if too thin."""
    if home_state.games < MIN_GAMES_BEFORE_USABLE or away_state.games < MIN_GAMES_BEFORE_USABLE:
        return None

    home_pct = home_state.win_pct
    away_pct = away_state.win_pct
    if home_pct is None or away_pct is None:
        return None

    home_split = home_state.split_pct(at_home=True)
    away_split = away_state.split_pct(at_home=False)
    game_date = (game.get("startDate") or "")[:10]

    features: dict[str, Any] = {
        "league": league,
        "recordDiff": round(home_pct - away_pct, 4),
        "splitDiff": (
            round(home_split - away_split, 4)
            if home_split is not None and away_split is not None
            else None
        ),
        "homeRest": _rest_days(home_state, game_date),
        "awayRest": _rest_days(away_state, game_date),
        # Static properties of the fixture, safe to compute at any time.
        "parkEdge": (
            park_run_environment(game.get("homeTeam"), game.get("venueName")) or {}
        ).get("edge")
        if league == "mlb"
        else None,
        "travelDiff": travel_edge(game.get("homeTeam"), game.get("awayTeam"))
        if league == "mlb"
        else None,
    }

    home_form = home_state.form_pct()
    away_form = away_state.form_pct()
    if home_form is not None and away_form is not None:
        features["formDiff"] = round(home_form - away_form, 4)

    return features


def replay_league(
    league: str,
    start: str,
    end: str,
    *,
    verify_ssl: bool = True,
    fetch=fetch_scoreboard,
    parse=parse_scoreboard,
) -> list[dict[str, Any]]:
    """Walk a date range forwards, emitting one labelled row per completed game.

    Forward order is not an implementation detail -- it is what makes the
    features pre-game. Each row is built from state accumulated strictly before
    that game, and only then is the result folded in.
    """
    states: dict[str, TeamState] = defaultdict(TeamState)
    rows: list[dict[str, Any]] = []

    current = date.fromisoformat(start)
    finish = date.fromisoformat(end)
    while current <= finish:
        iso = current.isoformat()
        try:
            games = parse(fetch(league, iso, verify_ssl=verify_ssl), league=league)
        except Exception as exc:  # noqa: BLE001 - one bad day must not stop the replay
            print(f"  {league} {iso}: skipped ({exc})")
            current += timedelta(days=1)
            continue

        for game in games:
            scores = _final_scores(game)
            if scores is None:
                continue
            home_score, away_score = scores
            home_team = game.get("homeTeam")
            away_team = game.get("awayTeam")
            if not home_team or not away_team:
                continue

            home_state = states[home_team]
            away_state = states[away_team]

            features = build_features(game, home_state, away_state, league=league)
            if features is not None:
                _assert_no_leakage(features, home_score, away_score)
                rows.append(
                    {
                        "eventId": str(game.get("eventId") or ""),
                        "league": league,
                        "date": iso,
                        "matchup": game.get("matchup"),
                        "homeWon": 1 if home_score > away_score else 0,
                        "features": features,
                    }
                )

            # Only now does the result enter team state, so the next game sees
            # it and this one never did.
            home_won = home_score > away_score
            played = (game.get("startDate") or iso)[:10]
            home_state.record(home_won, at_home=True, played_on=played)
            away_state.record(not home_won, at_home=False, played_on=played)

        current += timedelta(days=1)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="mlb")
    parser.add_argument("--start", required=True, help="ISO date, inclusive")
    parser.add_argument("--end", required=True, help="ISO date, inclusive")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification")
    parser.add_argument("--data-dir", default=str(ROOT / "docs" / "data"))
    args = parser.parse_args()

    print(f"Replaying {args.league} from {args.start} to {args.end}...")
    rows = replay_league(args.league, args.start, args.end, verify_ssl=not args.insecure)

    data_dir = Path(args.data_dir)
    path = data_dir / HISTORY_FILE
    existing = {}
    if path.is_file():
        try:
            existing = {row["eventId"]: row for row in json.loads(path.read_text())["rows"]}
        except (json.JSONDecodeError, KeyError, OSError):
            existing = {}
    for row in rows:
        existing[row["eventId"]] = row

    data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"rows": sorted(existing.values(), key=lambda r: r["date"])}, indent=2),
        encoding="utf-8",
    )

    home_wins = sum(row["homeWon"] for row in rows)
    print(f"  {len(rows)} usable games this run; {len(existing)} total in {path.name}")
    if rows:
        print(f"  home win rate {home_wins / len(rows) * 100:.1f}%")
    print("\nNo odds are available for past games, so these rows carry no")
    print("marketLogit. They screen candidates against the standalone model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
