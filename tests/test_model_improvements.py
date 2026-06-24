"""Tests for model accuracy improvements (calibration, coverage, dedupe, grading)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration_params import (
    MIN_PICK_CONFIDENCE,
    calibrate_probability,
    compute_calibration_params,
    is_publishable_pick,
)
from data_coverage import coverage_warnings, summarize_coverage
from data_providers.league_metrics import league_metrics_logit_adjustment
from data_providers.mlb_pitcher import mlb_pitching_logit_adjustment
from espn_client import parse_scoreboard
from mlb_predictions import (
    MARKET_BLEND_WEIGHT,
    _advanced_logit_adjustment,
    _lineup_logit_adjustment,
    apply_predictions,
    compute_true_probabilities,
)
from scripts.backtest_model import replay_snapshot, summarize_predictions
from sports_config import get_league

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ESPN_FIXTURE = FIXTURES / "espn_scoreboard_20260616.json"


class CalibrationParamsTests(unittest.TestCase):
    def test_strong_bucket_shrinks_more_than_default(self) -> None:
        params = {
            "defaultShrink": 0.88,
            "buckets": {
                "mlb": {
                    "strong_68+": 0.70,
                    "lean_57+": 0.84,
                    "coin_<57": 0.95,
                }
            },
        }
        strong = calibrate_probability(0.9, league="mlb", confidence_pct=72.0, params=params)
        baseline = calibrate_probability(0.9, league="mlb", confidence_pct=72.0, params={"buckets": {"mlb": {"strong_68+": 0.88, "lean_57+": 0.88, "coin_<57": 0.88}}})
        self.assertLess(strong, baseline)

    def test_compute_calibration_params_from_report(self) -> None:
        report = {
            "summary": {"graded": 10},
            "calibration": [
                {
                    "confidenceRange": "85-89",
                    "picks": 8,
                    "avgPredictedPct": 87.0,
                    "actualWinPct": 62.5,
                }
            ],
        }
        params = compute_calibration_params(report)
        self.assertIn("buckets", params)
        self.assertIn("default", params["buckets"])

    def test_publishable_pick_threshold(self) -> None:
        self.assertFalse(is_publishable_pick({"predictedWinner": "Team A", "confidence": 56.9}))
        self.assertTrue(is_publishable_pick({"predictedWinner": "Team A", "confidence": MIN_PICK_CONFIDENCE}))


class CoverageMetricsTests(unittest.TestCase):
    def test_summarize_coverage_counts_predictor(self) -> None:
        games = [
            {
                "enrichment": {"espnPredictorHome": 55.0, "espnPredictorAway": 45.0},
                "homeLineup": {"batters": [{"order": 1}]},
            },
            {"enrichment": {}},
        ]
        summary = summarize_coverage(games)
        self.assertEqual(summary["counts"]["espnPredictor"], 1)
        self.assertEqual(summary["pct"]["espnPredictor"], 50.0)

    def test_coverage_warning_when_predictor_low(self) -> None:
        warnings = coverage_warnings({"mlb": {"gameCount": 10, "pct": {"espnPredictor": 5.0}, "counts": {"espnPredictor": 0}}})
        self.assertEqual(len(warnings), 1)


class DedupeLogitTests(unittest.TestCase):
    def test_mlb_advanced_skips_ops_and_era(self) -> None:
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.8, "era": 3.5, "powerRating": 0.6},
            "awayAdvanced": {"opsProxy": 0.7, "era": 4.5, "powerRating": 0.5},
        }
        mlb_adj = _advanced_logit_adjustment(enrichment, "mlb")
        other_adj = _advanced_logit_adjustment(enrichment, "nfl")
        self.assertLess(abs(mlb_adj), abs(other_adj))

    def test_mlb_league_metrics_only_use_run_diff(self) -> None:
        enrichment = {
            "leagueMetrics": {
                "runDiffEdge": 20,
                "opsEdge": 0.1,
                "eraEdge": 1.0,
            }
        }
        adjustment = league_metrics_logit_adjustment(enrichment, "mlb")
        self.assertAlmostEqual(adjustment, 0.04, places=2)

    def test_mlb_pitching_adjustment_is_bullpen_only(self) -> None:
        enrichment = {
            "mlbPitching": {
                "homeBullpenEra": 3.5,
                "awayBullpenEra": 4.5,
                "homePitcherApiEra": 2.5,
                "awayPitcherApiEra": 5.5,
            }
        }
        adjustment = mlb_pitching_logit_adjustment({}, enrichment)
        self.assertAlmostEqual(adjustment, 0.06, places=2)


class MarketWeightTests(unittest.TestCase):
    def test_nfl_market_weight_higher_than_mlb(self) -> None:
        self.assertGreater(MARKET_BLEND_WEIGHT["nfl"], MARKET_BLEND_WEIGHT["mlb"])

    def test_compute_true_probabilities_uses_league_market_weight(self) -> None:
        lines = [
            {
                "sportsbook": "Test",
                "viewType": "MoneyLine",
                "currentLine": {"home": -200, "away": 170},
            }
        ]
        mlb_probs = compute_true_probabilities(
            model_home=0.55,
            enrichment={},
            league_config=get_league("mlb"),
            league="mlb",
            lines=lines,
        )
        nfl_probs = compute_true_probabilities(
            model_home=0.55,
            enrichment={},
            league_config=get_league("nfl"),
            league="nfl",
            lines=lines,
        )
        self.assertNotEqual(mlb_probs["homePct"], nfl_probs["homePct"])


class ApplyPredictionsTests(unittest.TestCase):
    def test_suppresses_coin_flip_picks(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "A",
            "awayTeam": "B",
            "homeRecord": "40-40",
            "awayRecord": "40-40",
            "enrichment": {},
        }
        with patch("mlb_predictions.predict_game", return_value={"predictedWinner": "A", "confidence": 52.0, "outcomeLabel": "A to win"}):
            result = apply_predictions([game])
        self.assertFalse(result[0]["prediction"].get("publishable"))
        self.assertIsNone(result[0].get("predictionRank"))

    def test_apply_predictions_only_ranks_publishable(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")
        ranked = apply_predictions(games)
        publishable = sorted(
            (game for game in ranked if game.get("predictionRank") is not None),
            key=lambda game: game["predictionRank"],
        )
        confidences = [game["prediction"]["confidence"] for game in publishable]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        if publishable:
            self.assertEqual(publishable[0]["predictionRank"], 1)


class LineupQualityTests(unittest.TestCase):
    def test_lineup_quality_prefers_stronger_ops_profile(self) -> None:
        game = {
            "homeLineup": {"batters": [{"order": 1}, {"order": 2}, {"order": 3}]},
            "awayLineup": {"batters": [{"order": 1}]},
        }
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.78},
            "awayAdvanced": {"opsProxy": 0.68},
        }
        self.assertGreater(_lineup_logit_adjustment(game, "mlb", enrichment), 0)


class BacktestHarnessTests(unittest.TestCase):
    def test_summarize_predictions_includes_calibration_params(self) -> None:
        data_dir = Path(__file__).resolve().parents[1] / "docs" / "data"
        if not (data_dir / "predictions_log.json").is_file():
            self.skipTest("predictions log not present")
        report = summarize_predictions(data_dir)
        self.assertIn("calibrationParams", report)

    def test_replay_snapshot_structure(self) -> None:
        data_dir = Path(__file__).resolve().parents[1] / "docs" / "data"
        snapshots = sorted(data_dir.glob("mlb_*.json"))
        if not snapshots:
            self.skipTest("no snapshot fixtures")
        schedule_date = snapshots[0].stem.split("_", 1)[1]
        report = replay_snapshot(data_dir, league="mlb", schedule_date=schedule_date)
        self.assertEqual(report["league"], "mlb")
        self.assertIn("gamesReplayed", report)


class EspnClientPitcherTests(unittest.TestCase):
    def test_parse_probable_includes_player_id(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")
        pitcher = games[0].get("homePitcher") or games[0].get("awayPitcher")
        self.assertIsNotNone(pitcher)
        self.assertIn("playerId", pitcher)


if __name__ == "__main__":
    unittest.main()
