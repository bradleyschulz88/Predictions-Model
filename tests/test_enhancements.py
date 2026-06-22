"""Tests for accuracy tracking and prediction enhancements."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from accuracy_tracker import grade_predictions, record_predictions
from mlb_predictions import confidence_label, extract_total_line, predict_game
from sports_config import list_league_ids


class PredictionEnhancementTests(unittest.TestCase):
    def test_confidence_labels(self) -> None:
        self.assertEqual(confidence_label(70), "Strong pick")
        self.assertEqual(confidence_label(58), "Lean")
        self.assertEqual(confidence_label(52), "Coin flip")

    def test_extract_total_line(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-110)"}}]
        self.assertEqual(extract_total_line(lines), 8.5)

    def test_predict_game_includes_model_fields(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeRecord": "30-20",
            "awayRecord": "20-30",
            "enrichment": {"homeMajorInjuries": [], "awayMajorInjuries": [{"player": "X", "status": "Out"}]},
        }
        prediction = predict_game(game)
        self.assertIn("confidenceLabel", prediction)
        self.assertIn("dataSources", prediction)
        self.assertIn("features", prediction)
        self.assertIn("pick", prediction["probabilities"])
        self.assertIn("implied", prediction["probabilities"])
        self.assertIn("dataCoverage", prediction["features"])


class AccuracyTrackerTests(unittest.TestCase):
    def test_record_and_grade_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            record_predictions(data_dir, {"mlb": {"scheduleDate": "2026-06-16", "fetchedAt": "now", "games": []}})
            accuracy = grade_predictions(data_dir)
            self.assertIn("summary", accuracy)
            self.assertTrue((data_dir / "predictions_log.json").is_file())

    def test_record_predictions_stores_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = {
                "league": "mlb",
                "scheduleDate": "2026-06-16",
                "fetchedAt": "now",
                "games": [
                    {
                        "eventId": "1",
                        "matchup": "A @ B",
                        "prediction": {
                            "predictedWinner": "B",
                            "predictedSide": "home",
                            "outcomeLabel": "B to win",
                            "confidence": 62.0,
                            "features": {"recordDiff": 0.1, "league": "mlb"},
                        },
                    }
                ],
            }
            record_predictions(data_dir, [payload])
            log = json.loads((data_dir / "predictions_log.json").read_text(encoding="utf-8"))
            self.assertEqual(log["predictions"]["1"]["features"]["recordDiff"], 0.1)


class LeagueConfigTests(unittest.TestCase):
    def test_includes_new_leagues(self) -> None:
        leagues = set(list_league_ids())
        self.assertTrue({"mlb", "nfl", "nba", "wnba", "worldcup", "epl", "afl"}.issubset(leagues))


class BuildPagesTests(unittest.TestCase):
    def test_build_overview_sorts_top_picks(self) -> None:
        from scripts.build_pages_data import build_overview

        payloads = {
            "mlb": {
                "leagueLabel": "MLB",
                "scheduleDate": "2026-06-16",
                "gameCount": 2,
                "topPick": "Yankees ML",
                "games": [
                    {"matchup": "A @ B", "eventId": "1", "prediction": {"outcomeLabel": "B ML", "confidence": 62, "confidenceLabel": "Lean"}},
                    {"matchup": "C @ D", "eventId": "2", "prediction": {"outcomeLabel": "C ML", "confidence": 71, "confidenceLabel": "Strong pick"}},
                ],
            },
            "nba": {
                "leagueLabel": "NBA",
                "scheduleDate": "2026-06-16",
                "gameCount": 1,
                "topPick": "Lakers ML",
                "games": [
                    {"matchup": "E @ F", "eventId": "3", "prediction": {"outcomeLabel": "F ML", "confidence": 68, "confidenceLabel": "Lean"}},
                ],
            },
        }
        overview = build_overview(payloads)
        self.assertEqual(len(overview["leagues"]), 2)
        self.assertGreaterEqual(overview["topPicksOverall"][0]["confidence"], overview["topPicksOverall"][1]["confidence"])

    def test_include_enrichment_for_all_dates(self) -> None:
        from scripts.build_pages_data import include_enrichment_for_date

        self.assertTrue(include_enrichment_for_date("2026-06-16", "2026-06-16"))
        self.assertTrue(include_enrichment_for_date("2026-06-13", "2026-06-16"))
        self.assertTrue(include_enrichment_for_date("2026-06-20", "2026-06-16"))

    def test_build_league_payload_resilient_retries_on_ssl(self) -> None:
        from unittest.mock import patch

        from scripts.build_pages_data import build_league_payload_resilient

        calls: list[bool] = []

        def fake_build(*_args, verify_ssl=True, **_kwargs):
            calls.append(verify_ssl)
            if verify_ssl:
                raise RuntimeError("SSL: CERTIFICATE_VERIFY_FAILED")
            return {"gameCount": 1, "games": [{"eventId": "1", "prediction": {"outcomeLabel": "Test"}}]}

        with patch("scripts.build_pages_data.build_league_payload", side_effect=fake_build):
            payload = build_league_payload_resilient(
                "mlb",
                "2026-06-21",
                include_enrichment=True,
                include_odds=False,
            )

        self.assertEqual(calls, [True, False])
        self.assertEqual(payload["gameCount"], 1)


if __name__ == "__main__":
    unittest.main()
