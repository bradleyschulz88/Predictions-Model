"""Tests for schedule, league metrics, and backtest helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_providers.league_metrics import enrich_league_metrics, soccer_draw_probability
from data_providers.schedule_advanced import compute_schedule_flags, schedule_flags_logit_adjustment
from scripts.backtest_model import summarize_predictions


class ScheduleAdvancedTests(unittest.TestCase):
    def test_back_to_back_detection(self) -> None:
        games = [
            {"homeTeam": "A", "awayTeam": "B", "startDate": "2026-06-14T23:00:00Z"},
            {"homeTeam": "C", "awayTeam": "A", "startDate": "2026-06-15T23:30:00Z"},
        ]
        flags = compute_schedule_flags(games, "A", "2026-06-16T00:00:00Z")
        self.assertTrue(flags.get("backToBack"))

    def test_schedule_flags_logit_penalizes_tired_home_team(self) -> None:
        enrichment = {
            "homeScheduleFlags": {"backToBack": True},
            "awayScheduleFlags": {"backToBack": False},
        }
        self.assertLess(schedule_flags_logit_adjustment(enrichment), 0)


class LeagueMetricsTests(unittest.TestCase):
    def test_nfl_efficiency_metrics(self) -> None:
        metrics = enrich_league_metrics(
            {},
            league="nfl",
            home_profile={"pointsPerGame": 28.0, "goalsAgainstPerGame": 20.0},
            away_profile={"pointsPerGame": 21.0, "goalsAgainstPerGame": 24.0},
        )
        self.assertGreater(metrics["efficiencyEdge"], 0)

    def test_soccer_draw_probability_uses_league_base(self) -> None:
        draw = soccer_draw_probability(
            league="epl",
            home_true=0.42,
            away_true=0.38,
            enrichment={"leagueMetrics": {"drawBaseRate": 0.25, "goalDiffEdge": 1}},
        )
        self.assertGreaterEqual(draw, 0.08)
        self.assertLessEqual(draw, 0.32)


class BacktestModelTests(unittest.TestCase):
    def test_summarize_predictions_handles_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            report = summarize_predictions(data_dir)
            self.assertEqual(report["summary"]["graded"], 0)


if __name__ == "__main__":
    unittest.main()
