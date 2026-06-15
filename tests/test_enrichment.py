"""Tests for ESPN enrichment and prediction reasoning."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from espn_enrichment import enrich_game, parse_event_enrichment
from espn_client import parse_scoreboard
from mlb_predictions import predict_game

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SUMMARY_FIXTURE = FIXTURES / "espn_summary_401815776.json"
SCOREBOARD_FIXTURE = FIXTURES / "espn_scoreboard_20260616.json"


class EnrichmentTests(unittest.TestCase):
    def test_parse_event_enrichment_from_fixture(self) -> None:
        with open(SUMMARY_FIXTURE, encoding="utf-8") as handle:
            summary = json.load(handle)

        enrichment = parse_event_enrichment(
            summary,
            home_team="Philadelphia Phillies",
            away_team="Miami Marlins",
        )

        self.assertAlmostEqual(enrichment["espnPredictorHome"], 52.9, places=1)
        self.assertEqual(enrichment["homeLastFive"]["record"], "2-3")
        self.assertTrue(enrichment["seasonSeries"]["summary"])
        self.assertTrue(enrichment["weather"])
        self.assertIn("homeLineup", enrichment)
        self.assertIn("homeMajorInjuries", enrichment)
        self.assertTrue(enrichment["awayMajorInjuries"])

    def test_predict_game_includes_why_they_win_with_enrichment(self) -> None:
        with open(SCOREBOARD_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")
        with open(SUMMARY_FIXTURE, encoding="utf-8") as handle:
            summary = json.load(handle)

        game = games[0]
        enrich_game(game, summary_fixture=summary)
        prediction = predict_game(game)

        self.assertTrue(prediction["whyTheyWin"])
        self.assertGreaterEqual(len(prediction["reasons"]), 2)
        self.assertIn("ESPN", " ".join(prediction["dataSources"]))
        self.assertTrue(game.get("homeMajorInjuries") is not None)
        self.assertTrue(game.get("homeLineup"))


if __name__ == "__main__":
    unittest.main()
