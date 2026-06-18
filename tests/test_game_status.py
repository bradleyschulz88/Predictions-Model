"""Tests for ESPN game status normalization."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from espn_client import fetch_scoreboard, parse_scoreboard
from game_status import normalize_espn_status


class GameStatusTests(unittest.TestCase):
    def test_postponed_is_not_final_or_live(self) -> None:
        flags = normalize_espn_status(
            {
                "name": "STATUS_POSTPONED",
                "state": "post",
                "completed": False,
                "description": "Postponed",
                "detail": "Postponed",
            },
            attendance=0,
            home_score=0,
            away_score=0,
        )
        self.assertTrue(flags["isPostponed"])
        self.assertTrue(flags["isVoided"])
        self.assertFalse(flags["isLive"])
        self.assertFalse(flags["isFinal"])

    def test_final_game(self) -> None:
        flags = normalize_espn_status(
            {
                "name": "STATUS_FINAL",
                "state": "post",
                "completed": True,
                "description": "Final",
                "detail": "Final",
            },
            attendance=32000,
            home_score=3,
            away_score=4,
        )
        self.assertTrue(flags["isFinal"])
        self.assertFalse(flags["isLive"])

    def test_stale_scoreless_in_progress_with_zero_attendance_is_not_live(self) -> None:
        started = datetime.now(timezone.utc) - timedelta(minutes=12)
        flags = normalize_espn_status(
            {
                "name": "STATUS_IN_PROGRESS",
                "state": "in",
                "completed": False,
                "description": "In Progress",
                "detail": "End 1st",
            },
            start_date=started.isoformat().replace("+00:00", "Z"),
            attendance=0,
            home_score=0,
            away_score=0,
        )
        self.assertFalse(flags["isLive"])
        self.assertTrue(flags["isWashedOut"])
        self.assertTrue(flags["isVoided"])
        self.assertEqual(flags["gameStatusText"], "Washed out")

    def test_low_score_attendance_zero_after_twenty_minutes_is_washed_out(self) -> None:
        started = datetime.now(timezone.utc) - timedelta(minutes=22)
        flags = normalize_espn_status(
            {
                "name": "STATUS_IN_PROGRESS",
                "state": "in",
                "completed": False,
                "description": "In Progress",
                "detail": "Top 2nd",
            },
            start_date=started.isoformat().replace("+00:00", "Z"),
            attendance=0,
            home_score=0,
            away_score=1,
        )
        self.assertFalse(flags["isLive"])
        self.assertTrue(flags["isWashedOut"])
        self.assertTrue(flags["isVoided"])
        self.assertEqual(flags["gameStatusText"], "Washed out")

    def test_stale_in_progress_phantom_score_is_washed_out(self) -> None:
        started = datetime.now(timezone.utc) - timedelta(minutes=25)
        flags = normalize_espn_status(
            {
                "name": "STATUS_IN_PROGRESS",
                "state": "in",
                "completed": False,
                "description": "In Progress",
                "detail": "Top 3rd",
            },
            start_date=started.isoformat().replace("+00:00", "Z"),
            attendance=0,
            home_score=1,
            away_score=1,
        )
        self.assertFalse(flags["isLive"])
        self.assertTrue(flags["isWashedOut"])
        self.assertTrue(flags["isVoided"])
        self.assertEqual(flags["gameStatusText"], "Washed out")

    def test_chicago_white_sox_fixture_not_live_with_zero_attendance(self) -> None:
        scoreboard = fetch_scoreboard("mlb", "2026-06-18", verify_ssl=False)
        games = parse_scoreboard(scoreboard, league="mlb")
        game = next(g for g in games if g["eventId"] == "401815803")
        self.assertFalse(game["isLive"])
        self.assertEqual(game["attendance"], 0)
        if game["isWashedOut"]:
            self.assertTrue(game["isVoided"])
            self.assertEqual(game["gameStatusText"], "Washed out")

    def test_active_in_progress_with_scoring_stays_live(self) -> None:
        started = datetime.now(timezone.utc) - timedelta(minutes=35)
        flags = normalize_espn_status(
            {
                "name": "STATUS_IN_PROGRESS",
                "state": "in",
                "completed": False,
                "description": "In Progress",
                "detail": "Bottom 2nd",
            },
            start_date=started.isoformat().replace("+00:00", "Z"),
            attendance=0,
            home_score=1,
            away_score=2,
        )
        self.assertTrue(flags["isLive"])

    def test_live_mlb_fixtures_from_espn(self) -> None:
        scoreboard = fetch_scoreboard("mlb", "2026-06-18", verify_ssl=False)
        games = parse_scoreboard(scoreboard, league="mlb")
        by_matchup = {game["matchup"]: game for game in games}

        postponed = by_matchup.get("San Francisco Giants @ Atlanta Braves")
        self.assertIsNotNone(postponed)
        assert postponed is not None
        self.assertTrue(postponed["isPostponed"])
        self.assertFalse(postponed["isLive"])
        self.assertFalse(postponed["isFinal"])

        final = by_matchup.get("Toronto Blue Jays @ Boston Red Sox")
        self.assertIsNotNone(final)
        assert final is not None
        self.assertTrue(final["isFinal"])
        self.assertFalse(final["isLive"])


if __name__ == "__main__":
    unittest.main()
