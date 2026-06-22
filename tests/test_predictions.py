"""Tests for win prediction model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from espn_client import parse_scoreboard
from mlb_data import fetch_dashboard_data
from mlb_predictions import (
    _injury_logit_adjustment,
    _lineup_logit_adjustment,
    _streak_logit_adjustment,
    apply_predictions,
    extract_prediction_features,
    predict_game,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ESPN_FIXTURE = FIXTURES / "espn_scoreboard_20260616.json"


class PredictionModelTests(unittest.TestCase):
    def test_predict_game_returns_percentages(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")
        prediction = predict_game(games[0])

        self.assertIn("predictedWinner", prediction)
        self.assertAlmostEqual(prediction["homeWinPct"] + prediction["awayWinPct"], 100.0, places=1)
        self.assertGreaterEqual(prediction["confidence"], 50.0)
        self.assertLessEqual(prediction["confidence"], 95.0)
        self.assertTrue(prediction["factors"])

    def test_apply_predictions_sorts_by_confidence_desc(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")

        ranked = apply_predictions(games)
        publishable = sorted(
            (game for game in ranked if game.get("prediction")),
            key=lambda game: game["predictionRank"],
        )
        confidences = [game["prediction"]["confidence"] for game in publishable]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        if publishable:
            self.assertEqual(publishable[0]["predictionRank"], 1)
            self.assertEqual(publishable[-1]["predictionRank"], len(publishable))

    def test_dashboard_payload_includes_predictions(self) -> None:
        payload = fetch_dashboard_data(fixture=ESPN_FIXTURE, include_odds=False, league="mlb")
        self.assertTrue(payload["topPick"])
        top_game = next(game for game in payload["games"] if game.get("predictionRank") == 1)
        self.assertIn("prediction", top_game)
        self.assertEqual(top_game["predictionRank"], 1)

    def test_weighted_injury_adjustment_prefers_qb_injury(self) -> None:
        enrichment = {
            "homeMajorInjuries": [{"player": "Starting QB", "status": "Out", "detail": "quarterback"}],
            "awayMajorInjuries": [],
        }
        self.assertLess(_injury_logit_adjustment(enrichment, "nfl"), 0)
        flipped = {
            "homeMajorInjuries": [],
            "awayMajorInjuries": [{"player": "Starting QB", "status": "Out", "detail": "quarterback"}],
        }
        self.assertGreater(_injury_logit_adjustment(flipped, "nfl"), 0)

    def test_lineup_adjustment_uses_confirmed_starters(self) -> None:
        game = {
            "homeLineup": {"batters": [{"order": 1}, {"order": 2}, {"order": 3}]},
            "awayLineup": {"batters": [{"order": 1}]},
        }
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.75},
            "awayAdvanced": {"opsProxy": 0.70},
        }
        self.assertGreater(_lineup_logit_adjustment(game, "mlb", enrichment), 0)

    def test_prediction_includes_feature_vector(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeRecord": "30-20",
            "awayRecord": "20-30",
            "enrichment": {},
        }
        prediction = predict_game(game)
        self.assertIn("features", prediction)
        self.assertEqual(prediction["features"]["league"], "mlb")
        self.assertIn("recordDiff", prediction["features"])


if __name__ == "__main__":
    unittest.main()
