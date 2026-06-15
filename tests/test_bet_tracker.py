"""Tests for bet tracker / pick grading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from accuracy_tracker import (
    american_odds_profit,
    extract_pick_american_odds,
    grade_predictions,
    record_predictions,
)


class BetTrackerTests(unittest.TestCase):
    def test_american_odds_profit(self) -> None:
        self.assertAlmostEqual(american_odds_profit(-150, True), 0.667, places=2)
        self.assertEqual(american_odds_profit(130, True), 1.3)
        self.assertEqual(american_odds_profit(-150, False), -1.0)

    def test_extract_pick_american_odds(self) -> None:
        game = {
            "lines": [
                {
                    "viewType": "MoneyLine",
                    "currentLine": {"homeOdds": -140, "awayOdds": 120},
                }
            ]
        }
        self.assertEqual(extract_pick_american_odds(game, "home"), -140)
        self.assertEqual(extract_pick_american_odds(game, "away"), 120)

    def test_record_predictions_stores_pick_odds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = {
                "league": "mlb",
                "scheduleDate": "2026-06-15",
                "fetchedAt": "now",
                "games": [
                    {
                        "eventId": "123",
                        "matchup": "A @ B",
                        "homeTeam": "B",
                        "awayTeam": "A",
                        "lines": [
                            {
                                "viewType": "MoneyLine",
                                "currentLine": {"homeOdds": -130, "awayOdds": 110},
                            }
                        ],
                        "prediction": {
                            "predictedWinner": "B",
                            "predictedSide": "home",
                            "outcomeLabel": "B to win",
                            "confidence": 62,
                        },
                    }
                ],
            }
            record_predictions(data_dir, [payload])
            log = (data_dir / "predictions_log.json").read_text(encoding="utf-8")
            self.assertIn('"pickOdds": -130', log)

    def test_grade_predictions_builds_pick_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            record_predictions(
                data_dir,
                [
                    {
                        "league": "mlb",
                        "scheduleDate": "2026-06-15",
                        "fetchedAt": "now",
                        "games": [
                            {
                                "eventId": "999",
                                "matchup": "Away @ Home",
                                "prediction": {
                                    "predictedWinner": "Home",
                                    "predictedSide": "home",
                                    "outcomeLabel": "Home to win",
                                    "confidence": 60,
                                },
                            }
                        ],
                    }
                ],
            )
            accuracy = grade_predictions(data_dir, verify_ssl=True)
            self.assertIn("picksByEventId", accuracy)
            self.assertIn("pendingPicks", accuracy)
            self.assertIn("summary", accuracy)
            self.assertIn("streak", accuracy["summary"])


if __name__ == "__main__":
    unittest.main()
