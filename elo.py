"""Elo ratings from graded results, as a strength measure that knows who you played.

The model's strength feature averages season win rate, home/road splits and a
power rating. All three are schedule-blind: a 60-40 team that feasted on weak
opponents scores the same as a 60-40 team that did it against contenders.

Elo is not schedule-blind. Every update is proportional to the opponent's
rating, so beating a strong side moves you further than beating a weak one, and
the whole league's ratings stay on one comparable scale. It also updates game by
game rather than waiting for a season aggregate to shift.

Ratings are rebuilt from the graded history on every run rather than stored, so
there is no state to drift and no risk of a rating file disagreeing with the
results it claims to summarise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

START_RATING = 1500.0

# K controls how fast ratings move. Baseball is mostly noise -- even the best
# team loses 35% of the time -- so a single result should barely move a rating.
# Basketball and football carry far more signal per game.
LEAGUE_K = {
    "mlb": 4.0,
    "nba": 20.0,
    "wnba": 20.0,
    "nfl": 20.0,
    "afl": 18.0,
    "epl": 20.0,
}
DEFAULT_K = 15.0

# Home advantage expressed in rating points, which is how Elo wants it. These
# are starting values only; the fitted model still carries its own home-field
# intercept, so this just needs to be roughly right.
LEAGUE_HOME_ADVANTAGE = {
    "mlb": 24.0,
    "nba": 60.0,
    "wnba": 55.0,
    "nfl": 55.0,
    "afl": 45.0,
    "epl": 60.0,
}
DEFAULT_HOME_ADVANTAGE = 45.0

RATINGS_FILE = "elo_ratings.json"


def expected_score(rating: float, opponent: float) -> float:
    """Elo's own win-probability estimate for a rating gap."""
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400.0))


def margin_multiplier(margin: int, rating_gap: float) -> float:
    """Dampen blowouts, and dampen them harder when the favourite wins big.

    Without this a 20-run drubbing moves a rating as if it were twenty separate
    wins. The rating-gap term is the standard autocorrelation correction: a
    strong side beating a weak one by a lot is expected, not informative.
    """
    return math.log(abs(margin) + 1.0) * (2.2 / (rating_gap * 0.001 + 2.2))


# A team's season record, converted to rating points, as the starting estimate.
# Elo from a cold 1500 needs a season to learn who is good; seeding it from the
# record it already has means it starts roughly right and spends its updates on
# the thing records cannot see -- who you actually played.
SEED_SCALE = 400.0


class EloTable:
    """Ratings for one league, advanced one game at a time."""

    def __init__(self, league: str, seeds: dict[str, float] | None = None) -> None:
        self.league = league
        self.k = LEAGUE_K.get(league, DEFAULT_K)
        self.home_advantage = LEAGUE_HOME_ADVANTAGE.get(league, DEFAULT_HOME_ADVANTAGE)
        self.ratings: dict[str, float] = {}
        self.games_seen: dict[str, int] = {}
        self.seeds = seeds or {}

    def rating(self, team: str) -> float:
        if team in self.ratings:
            return self.ratings[team]
        seed = self.seeds.get(team)
        if seed is None:
            return START_RATING
        # win rate 0.5 -> 1500, 0.6 -> 1540, 0.4 -> 1460
        return START_RATING + (seed - 0.5) * SEED_SCALE

    def pregame_edge(self, home: str, away: str) -> float:
        """Home rating minus away, before this game is played, including home field."""
        return self.rating(home) - self.rating(away) + self.home_advantage

    def observe(self, home: str, away: str, home_score: int, away_score: int) -> None:
        home_rating, away_rating = self.rating(home), self.rating(away)
        expected_home = expected_score(home_rating + self.home_advantage, away_rating)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        margin = abs(home_score - away_score)
        gap = (home_rating + self.home_advantage - away_rating) * (1 if actual_home else -1)
        multiplier = margin_multiplier(margin, gap) if margin else 1.0

        shift = self.k * multiplier * (actual_home - expected_home)
        self.ratings[home] = home_rating + shift
        self.ratings[away] = away_rating - shift
        self.games_seen[home] = self.games_seen.get(home, 0) + 1
        self.games_seen[away] = self.games_seen.get(away, 0) + 1


def parse_matchup(matchup: str | None) -> tuple[str, str] | None:
    """'Away Team @ Home Team' -> (home, away). None if it does not parse."""
    if not matchup or " @ " not in str(matchup):
        return None
    away, _, home = str(matchup).partition(" @ ")
    away, home = away.strip(), home.strip()
    if not away or not home:
        return None
    return home, away


def _score(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def results_from_accuracy(accuracy: dict[str, Any]) -> list[dict[str, Any]]:
    """Chronological graded results, the only input Elo needs."""
    rows: list[dict[str, Any]] = []
    for event_id, record in (accuracy.get("picksByEventId") or {}).items():
        if record.get("status") != "graded":
            continue
        teams = parse_matchup(record.get("matchup"))
        home_score = _score(record.get("homeScore"))
        away_score = _score(record.get("awayScore"))
        if not teams or home_score is None or away_score is None:
            continue
        rows.append(
            {
                "eventId": str(event_id),
                "league": record.get("league") or "unknown",
                "date": record.get("scheduleDate") or record.get("date") or "",
                "home": teams[0],
                "away": teams[1],
                "homeScore": home_score,
                "awayScore": away_score,
            }
        )
    rows.sort(key=lambda row: (row["date"], row["eventId"]))
    return rows


def build_history(
    results: Iterable[dict[str, Any]],
    seeds: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, EloTable], dict[str, float]]:
    """Replay results in order, recording each game's rating gap *before* it was played.

    Returning the pre-game edge is the whole point: using a rating that already
    contains the game's own result would leak the answer into the feature.
    """
    tables: dict[str, EloTable] = {}
    pregame: dict[str, float] = {}

    for row in results:
        league = row["league"]
        table = tables.setdefault(league, EloTable(league, (seeds or {}).get(league)))
        pregame[row["eventId"]] = table.pregame_edge(row["home"], row["away"])
        table.observe(row["home"], row["away"], row["homeScore"], row["awayScore"])

    return tables, pregame


def load_ratings(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    path = data_dir / RATINGS_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def rating_edge(ratings: dict[str, Any], league: str, home: str | None, away: str | None) -> float | None:
    """Pure pre-game rating gap between two clubs, from a stored ratings table.

    Deliberately excludes home advantage, unlike ``EloTable.pregame_edge``.
    The two have different jobs: pregame_edge feeds Elo's own expected-score
    update, where home field belongs, while this feeds the model's eloDiff
    feature, where it does not.

    Adding it here made eloDiff carry a per-league constant -- +24 rating
    points for MLB against +60 for NBA, so a 0.36 spread in eloDiff units --
    that had nothing to do with the two teams playing. Standardisation is
    global rather than per-league, so that constant survived as a league
    indicator smuggled into a feature meant to measure team strength, and
    leagueIntercepts already exists to carry exactly that. The fitted model
    also carries home field in its own intercept, so including it here
    counted the same effect twice.
    """
    table = (ratings.get("leagues") or {}).get(league)
    if not table or not home or not away:
        return None
    values = table.get("ratings") or {}
    # Only meaningful once both sides have a rating that has actually moved.
    if home not in values or away not in values:
        return None
    return float(values[home]) - float(values[away])


def build_and_write(data_dir: Path) -> dict[str, Any]:
    """Rebuild ratings from graded history and persist them for the next build."""
    accuracy_path = data_dir / "accuracy.json"
    try:
        accuracy = json.loads(accuracy_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        accuracy = {}

    results = results_from_accuracy(accuracy)
    tables, _ = build_history(results)

    payload = {
        "builtFrom": len(results),
        "leagues": {
            league: {
                "k": table.k,
                "homeAdvantage": table.home_advantage,
                "ratings": {team: round(value, 1) for team, value in sorted(table.ratings.items())},
                "gamesSeen": table.games_seen,
            }
            for league, table in sorted(tables.items())
        },
    }
    (data_dir / RATINGS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
