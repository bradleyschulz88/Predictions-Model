"""Tests for league timezone schedule dates."""

from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from schedule_dates import (
    default_game_date,
    get_schedule_timezone,
    league_schedule_date,
    schedule_dates_for_league,
)


class ScheduleDateTests(unittest.TestCase):
    def test_mlb_uses_eastern_timezone(self) -> None:
        self.assertEqual(get_schedule_timezone("mlb"), "America/New_York")

    def test_afl_uses_melbourne_timezone(self) -> None:
        self.assertEqual(get_schedule_timezone("afl"), "Australia/Melbourne")

    def test_default_mlb_date_uses_us_calendar_not_utc_tomorrow(self) -> None:
        # June 16 08:00 in Sydney = June 15 evening in New York -> MLB schedule day is June 15
        sydney_morning = datetime(2026, 6, 16, 8, 0, tzinfo=ZoneInfo("Australia/Sydney"))
        with patch("schedule_dates.league_now", return_value=sydney_morning.astimezone(ZoneInfo("America/New_York"))):
            self.assertEqual(default_game_date("mlb"), "2026-06-15")

    def test_default_mlb_before_10am_uses_yesterday(self) -> None:
        early_et = datetime(2026, 6, 16, 8, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("schedule_dates.league_now", return_value=early_et):
            self.assertEqual(default_game_date("mlb"), "2026-06-15")

    def test_default_mlb_afternoon_uses_today(self) -> None:
        afternoon_et = datetime(2026, 6, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("schedule_dates.league_now", return_value=afternoon_et):
            self.assertEqual(default_game_date("mlb"), "2026-06-15")

    def test_schedule_dates_include_yesterday_for_mlb(self) -> None:
        afternoon_et = datetime(2026, 6, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("schedule_dates.league_now", return_value=afternoon_et):
            dates = schedule_dates_for_league("mlb")
        self.assertIn("2026-06-15", dates)
        self.assertIn("2026-06-14", dates)
        self.assertIn("2026-06-16", dates)

    def test_schedule_dates_include_upcoming_week_for_mlb(self) -> None:
        afternoon_et = datetime(2026, 6, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("schedule_dates.league_now", return_value=afternoon_et):
            dates = schedule_dates_for_league("mlb")
        self.assertIn("2026-06-12", dates)
        self.assertIn("2026-06-22", dates)
        self.assertGreaterEqual(len(dates), 10)

    def test_league_schedule_date_format(self) -> None:
        self.assertRegex(league_schedule_date("epl"), r"^\d{4}-\d{2}-\d{2}$")

    def test_wnba_uses_eastern_timezone(self) -> None:
        self.assertEqual(get_schedule_timezone("wnba"), "America/New_York")

    def test_schedule_dates_include_yesterday_for_wnba(self) -> None:
        afternoon_et = datetime(2026, 6, 15, 18, 0, tzinfo=ZoneInfo("America/New_York"))
        with patch("schedule_dates.league_now", return_value=afternoon_et):
            dates = schedule_dates_for_league("wnba")
        self.assertIn("2026-06-15", dates)
        self.assertIn("2026-06-14", dates)


if __name__ == "__main__":
    unittest.main()
