"""Tests for schedule, league metrics, and backtest helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_providers.league_metrics import enrich_league_metrics, soccer_draw_probability
from data_providers.schedule_advanced import (
    clear_rolling_schedule_cache,
    compute_schedule_flags,
    fetch_rolling_schedule_games,
    schedule_flags_logit_adjustment,
)
from scripts.backtest_model import _fitted_walk_forward, summarize_predictions, write_calibration_report


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

    def test_mlb_league_metrics_include_run_and_ops_edges(self) -> None:
        metrics = enrich_league_metrics(
            {},
            league="mlb",
            home_profile={"runDifferential": 42, "opsProxy": 0.780, "era": 3.80},
            away_profile={"runDifferential": 10, "opsProxy": 0.720, "era": 4.20},
        )
        self.assertEqual(metrics["runDiffEdge"], 32)
        self.assertGreater(metrics["opsEdge"], 0)
        self.assertGreater(metrics["eraEdge"], 0)


class BacktestModelTests(unittest.TestCase):
    def test_summarize_predictions_handles_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            report = summarize_predictions(data_dir)
            self.assertEqual(report["summary"]["graded"], 0)

    def test_write_calibration_report_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            report = write_calibration_report(data_dir)
            self.assertTrue((data_dir / "calibration.json").is_file())
            self.assertIn("thresholds", report)


class RollingScheduleTests(unittest.TestCase):
    def test_fetch_rolling_schedule_merges_current_games(self) -> None:
        clear_rolling_schedule_cache()
        current = [
            {
                "eventId": "99",
                "homeTeam": "A",
                "awayTeam": "B",
                "startDate": "2026-06-16T00:00:00Z",
            }
        ]

        def fake_scoreboard(league: str, day: str, **kwargs):  # noqa: ANN001
            if day == "2026-06-15":
                return {
                    "events": [
                        {
                            "id": "1",
                            "date": "2026-06-15T23:00:00Z",
                            "competitions": [
                                {
                                    "date": "2026-06-15T23:00:00Z",
                                    "competitors": [
                                        {"homeAway": "away", "team": {"displayName": "A"}},
                                        {"homeAway": "home", "team": {"displayName": "C"}},
                                    ],
                                }
                            ],
                        }
                    ]
                }
            return {"events": []}

        from unittest.mock import patch

        with patch("data_providers.schedule_advanced.fetch_scoreboard", side_effect=fake_scoreboard):
            games = fetch_rolling_schedule_games(
                "mlb",
                "2026-06-16",
                lookback_days=1,
                current_games=current,
            )
        event_ids = {str(game.get("eventId")) for game in games}
        self.assertIn("99", event_ids)
        self.assertIn("1", event_ids)
        clear_rolling_schedule_cache()


class FittedWalkForwardTests(unittest.TestCase):
    """_fitted_walk_forward reads the shipped model's own l2 to reproduce its
    score honestly. model_fit.py's differential shrinkage made that l2 a dict
    ({"strengthDiff": ..., "marketLogit": ...}) rather than always a float --
    this pins that a shipped dict-valued l2 does not crash the evaluation
    report, which is exactly what a bare float(l2) call would do."""

    def test_a_dict_valued_shipped_l2_does_not_raise(self) -> None:
        from unittest.mock import MagicMock, patch

        import model_fit

        fake_model = MagicMock()
        fake_model.metadata = {"l2": {"strengthDiff": 30.0, "marketLogit": 1.0}}

        samples = [
            model_fit.Sample(
                values={"strengthDiff": 0.1, "marketLogit": 0.2},
                label=1,
                league="mlb",
                date=f"2026-06-{day:02d}",
            )
            for day in range(1, 29)
        ] * 2  # enough rows for walk_forward_scores to attempt folds

        with patch("scripts.backtest_model.model_fit.load_model", return_value=fake_model), \
             patch("scripts.backtest_model.model_fit.samples_from_log", return_value=(samples, 0.0)):
            result = _fitted_walk_forward(Path("unused"))

        self.assertEqual(result.get("l2"), {"strengthDiff": 30.0, "marketLogit": 1.0})

    def test_falls_back_to_1_0_when_no_model_is_shipped(self) -> None:
        from unittest.mock import patch

        import model_fit

        samples = [
            model_fit.Sample(
                values={"strengthDiff": 0.1, "marketLogit": 0.2},
                label=1,
                league="mlb",
                date=f"2026-06-{day:02d}",
            )
            for day in range(1, 29)
        ] * 2

        with patch("scripts.backtest_model.model_fit.load_model", return_value=None), \
             patch("scripts.backtest_model.model_fit.samples_from_log", return_value=(samples, 0.0)):
            result = _fitted_walk_forward(Path("unused"))

        self.assertEqual(result.get("l2"), 1.0)


if __name__ == "__main__":
    unittest.main()
