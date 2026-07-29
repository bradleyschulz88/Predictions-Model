"""Historical backfill -- and the leakage discipline that makes it worth having."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.backfill_history import (  # noqa: E402
    MIN_GAMES_BEFORE_USABLE,
    TeamState,
    _assert_no_leakage,
    _final_scores,
    build_features,
    replay_league,
)


def _game(home: str, away: str, hs: int, aws: int, iso: str, event_id: str) -> dict:
    return {
        "eventId": event_id, "homeTeam": home, "awayTeam": away,
        "homeScore": hs, "awayScore": aws, "startDate": f"{iso}T23:00Z",
        "matchup": f"{away} @ {home}", "isFinal": True,
        "isVoided": False, "isPostponed": False, "isCanceled": False,
    }


class SeasonFixture:
    """A deterministic season where the home side always wins by exactly 4.

    Chosen so any leak of the result is unmistakable: a margin of 4 or a score
    of 7 appearing in a feature could only have come from the game itself.
    """

    def __init__(self, days: int = 40) -> None:
        from datetime import date, timedelta

        self.by_date: dict[str, list[dict]] = {}
        start = date(2025, 4, 1)
        clubs = ["Colorado Rockies", "San Diego Padres", "New York Mets", "Seattle Mariners"]
        for i in range(days):
            iso = (start + timedelta(days=i)).isoformat()
            home, away = clubs[i % 4], clubs[(i + 1) % 4]
            self.by_date[iso] = [_game(home, away, 7, 3, iso, f"e{i}")]

    def fetch(self, league, iso, **kwargs):
        return {"_date": iso}

    def parse(self, payload, league=None):
        return self.by_date.get(payload["_date"], [])


class LeakageTests(unittest.TestCase):
    """The failure that makes a backfill look superb and perform terribly."""

    def test_a_feature_matching_the_result_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            _assert_no_leakage({"recordDiff": 4.0}, 7, 3)

    def test_ordinary_features_pass(self) -> None:
        _assert_no_leakage({"recordDiff": 0.25, "homeRest": 1}, 7, 3)

    def test_a_full_replay_never_leaks(self) -> None:
        """The guard runs on every row; this proves it never fires wrongly."""
        season = SeasonFixture()
        rows = replay_league("mlb", "2025-04-01", "2025-05-10",
                             fetch=season.fetch, parse=season.parse)
        self.assertTrue(rows)

    def test_state_is_read_before_the_result_is_recorded(self) -> None:
        """The ordering that makes the features pre-game.

        With the home side always winning, a club's record must never already
        include the game being predicted.
        """
        season = SeasonFixture()
        rows = replay_league("mlb", "2025-04-01", "2025-05-10",
                             fetch=season.fetch, parse=season.parse)
        # Each club plays every other day, so by the last row no club can have a
        # perfect record derived from a game it has not played yet.
        for row in rows:
            self.assertLessEqual(abs(row["features"]["recordDiff"]), 1.0)


class ReplayTests(unittest.TestCase):
    def test_thin_clubs_are_skipped(self) -> None:
        """A 2-0 record is not a signal; rows start once there is a base."""
        season = SeasonFixture(days=6)
        rows = replay_league("mlb", "2025-04-01", "2025-04-06",
                             fetch=season.fetch, parse=season.parse)
        self.assertEqual(rows, [], "no club has enough games yet")

    def test_rows_appear_once_clubs_have_history(self) -> None:
        season = SeasonFixture(days=60)
        rows = replay_league("mlb", "2025-04-01", "2025-05-30",
                             fetch=season.fetch, parse=season.parse)
        self.assertTrue(rows)
        self.assertTrue(all(row["homeWon"] == 1 for row in rows))

    def test_a_failed_day_does_not_stop_the_replay(self) -> None:
        season = SeasonFixture(days=60)

        def flaky(league, iso, **kwargs):
            if iso == "2025-04-20":
                raise RuntimeError("ESPN 500")
            return season.fetch(league, iso)

        rows = replay_league("mlb", "2025-04-01", "2025-05-30",
                             fetch=flaky, parse=season.parse)
        self.assertTrue(rows, "one bad day must not abort the season")

    def test_static_fixture_properties_are_populated(self) -> None:
        season = SeasonFixture(days=60)
        rows = replay_league("mlb", "2025-04-01", "2025-05-30",
                             fetch=season.fetch, parse=season.parse)
        row = rows[-1]
        self.assertIsNotNone(row["features"]["parkEdge"])
        self.assertIsNotNone(row["features"]["travelDiff"])

    def test_no_market_feature_is_invented(self) -> None:
        """Historical odds do not exist; fabricating them is worse than a gap."""
        season = SeasonFixture(days=60)
        rows = replay_league("mlb", "2025-04-01", "2025-05-30",
                             fetch=season.fetch, parse=season.parse)
        for row in rows:
            self.assertNotIn("impliedHome", row["features"])
            self.assertNotIn("marketLogit", row["features"])


class TeamStateTests(unittest.TestCase):
    def test_splits_are_tracked_separately(self) -> None:
        state = TeamState()
        state.record(True, at_home=True, played_on="2025-04-01")
        state.record(False, at_home=False, played_on="2025-04-02")
        self.assertEqual(state.split_pct(at_home=True), 1.0)
        self.assertEqual(state.split_pct(at_home=False), 0.0)

    def test_form_is_a_rolling_window(self) -> None:
        state = TeamState()
        for _ in range(8):
            state.record(True, at_home=True, played_on="2025-04-01")
        state.record(False, at_home=True, played_on="2025-04-02")
        self.assertAlmostEqual(state.form_pct(), 0.8, places=2)

    def test_an_unplayed_club_has_no_rate(self) -> None:
        self.assertIsNone(TeamState().win_pct)
        self.assertIsNone(TeamState().form_pct())


class ScoreParsingTests(unittest.TestCase):
    def test_unfinished_and_abandoned_games_are_excluded(self) -> None:
        base = _game("A", "B", 5, 3, "2025-04-01", "1")
        self.assertIsNotNone(_final_scores(base))
        for flag in ("isVoided", "isPostponed", "isCanceled"):
            spoiled = dict(base); spoiled[flag] = True
            self.assertIsNone(_final_scores(spoiled), flag)
        unfinished = dict(base); unfinished["isFinal"] = False
        self.assertIsNone(_final_scores(unfinished))

    def test_missing_scores_are_excluded(self) -> None:
        broken = _game("A", "B", 5, 3, "2025-04-01", "1")
        broken["homeScore"] = None
        self.assertIsNone(_final_scores(broken))


if __name__ == "__main__":
    unittest.main()
