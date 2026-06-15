"""Tests for win prediction model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from espn_client import parse_scoreboard
from mlb_data import fetch_dashboard_data
from mlb_predictions import apply_predictions, predict_game

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
        confidences = [game["prediction"]["confidence"] for game in ranked]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertEqual(ranked[0]["predictionRank"], 1)
        self.assertEqual(ranked[-1]["predictionRank"], len(ranked))

    def test_dashboard_payload_includes_predictions(self) -> None:
        payload = fetch_dashboard_data(fixture=ESPN_FIXTURE, include_odds=False, league="mlb")
        self.assertTrue(payload["topPick"])
        self.assertIn("prediction", payload["games"][0])
        self.assertEqual(payload["games"][0]["predictionRank"], 1)


if __name__ == "__main__":
    unittest.main()
