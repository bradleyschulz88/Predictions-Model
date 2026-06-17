"""Tests for implied and true probability calculations."""

from __future__ import annotations

import unittest

from mlb_predictions import (
    american_odds_to_implied,
    compute_implied_probabilities,
    compute_true_probabilities,
    predict_game,
)
from sports_config import get_league


class ImpliedProbabilityTests(unittest.TestCase):
    def test_american_odds_to_implied(self) -> None:
        self.assertAlmostEqual(american_odds_to_implied(-150), 0.6, places=2)
        self.assertAlmostEqual(american_odds_to_implied(130), 0.4348, places=3)

    def test_consensus_implied_across_books(self) -> None:
        lines = [
            {
                "sportsbook": "Book A",
                "viewType": "MoneyLine",
                "currentLine": {"home": -150, "away": 130},
            },
            {
                "sportsbook": "Book B",
                "viewType": "MoneyLine",
                "currentLine": {"home": -140, "away": 120},
            },
        ]
        implied = compute_implied_probabilities(lines)
        self.assertTrue(implied["available"])
        self.assertEqual(implied["booksUsed"], 2)
        self.assertAlmostEqual(
            implied["consensus"]["homePct"] + implied["consensus"]["awayPct"],
            100.0,
            places=0,
        )

    def test_sbr_moneyline_shape(self) -> None:
        lines = [
            {
                "sportsbook": "betmgm",
                "viewType": "MoneyLineDataOpeningAndLatestOddsDataView",
                "currentLine": {"homeOdds": -140, "awayOdds": 115, "drawOdds": 0},
            },
            {
                "sportsbook": "draftkings",
                "viewType": "MoneyLineDataOpeningAndLatestOddsDataView",
                "currentLine": {"homeOdds": -138, "awayOdds": 118, "drawOdds": 0},
            },
        ]
        implied = compute_implied_probabilities(lines)
        self.assertTrue(implied["available"])
        self.assertEqual(implied["booksUsed"], 2)
        self.assertGreater(implied["consensus"]["homePct"], 50.0)

    def test_true_probability_components(self) -> None:
        true_probs = compute_true_probabilities(
            model_home=0.58,
            enrichment={
                "espnPredictorHome": 60.0,
                "espnPredictorAway": 40.0,
                "homeAdvanced": {"powerRating": 0.62},
                "awayAdvanced": {"powerRating": 0.48},
                "homeLastFive": {"record": "4-1"},
                "awayLastFive": {"record": "2-3"},
            },
            league_config=get_league("mlb"),
        )
        self.assertIn("homePct", true_probs)
        self.assertGreater(true_probs["homePct"], 50.0)
        self.assertTrue(true_probs["components"])

    def test_predict_game_includes_probabilities(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeRecord": "50-30",
            "awayRecord": "35-45",
            "lines": [
                {
                    "sportsbook": "TestBook",
                    "viewType": "MoneyLine",
                    "currentLine": {"home": -160, "away": 140},
                }
            ],
            "enrichment": {
                "espnPredictorHome": 58.0,
                "espnPredictorAway": 42.0,
                "homeAdvanced": {"powerRating": 0.6},
                "awayAdvanced": {"powerRating": 0.45},
            },
        }
        prediction = predict_game(game)
        self.assertIn("probabilities", prediction)
        self.assertIn("true", prediction["probabilities"])
        self.assertIn("pick", prediction["probabilities"])
        self.assertIn("teamProbabilities", prediction)
        self.assertIn("home", prediction["teamProbabilities"])
        self.assertEqual(prediction["teamProbabilities"]["home"]["truePct"], prediction["probabilities"]["true"]["homePct"])


if __name__ == "__main__":
    unittest.main()
