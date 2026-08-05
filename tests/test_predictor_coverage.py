"""Predictor coverage must measure the feed, not the clock.

Every overnight build warned that ESPN predictor coverage was 0%, which read
as a broken feed and was not. ESPN publishes the Matchup Predictor before a
game and drops it the moment the game is final, so a slate whose games have
all been played cannot carry one. Verified live from a runner: the 2026-08-05
slate returned 100% coverage, the 2026-08-04 slate 0/15 with every game
'Final'.

The warning now counts only games that could still have a predictor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_coverage import coverage_warnings, summarize_coverage  # noqa: E402


def _predictor_warnings(summary: dict) -> list[str]:
    """Only the predictor warning. These fixtures have no lines and no
    predictions, so they legitimately trip the odds and publish-rate
    checks too, and those are a different question."""
    return [w for w in coverage_warnings({"mlb": summary}) if "predictor" in w]


def _game(*, predictor: bool, final: bool = False, voided: bool = False) -> dict:
    enrichment = {}
    if predictor:
        enrichment = {"espnPredictorHome": 60.5, "espnPredictorAway": 39.5}
    return {
        "league": "mlb",
        "isFinal": final,
        "isVoided": voided,
        "statusType": "STATUS_FINAL" if final else "STATUS_SCHEDULED",
        "enrichment": enrichment,
        "lines": [],
    }


class PredictorEligibilityTests(unittest.TestCase):
    def test_finished_slate_is_not_counted_against_the_feed(self) -> None:
        summary = summarize_coverage([_game(predictor=False, final=True) for _ in range(15)])
        self.assertEqual(summary["predictorEligible"], 0)
        self.assertIsNone(summary["predictorPct"])

    def test_finished_slate_raises_no_warning(self) -> None:
        summary = summarize_coverage([_game(predictor=False, final=True) for _ in range(15)])
        self.assertEqual(_predictor_warnings(summary), [])

    def test_upcoming_slate_with_predictors_is_full_coverage(self) -> None:
        summary = summarize_coverage([_game(predictor=True) for _ in range(15)])
        self.assertEqual(summary["predictorEligible"], 15)
        self.assertEqual(summary["predictorPct"], 100.0)
        self.assertEqual(_predictor_warnings(summary), [])

    def test_a_genuinely_missing_feed_still_warns(self) -> None:
        """The point is to keep the real signal, not to silence the check."""
        summary = summarize_coverage([_game(predictor=False) for _ in range(15)])
        warnings = _predictor_warnings(summary)
        self.assertEqual(len(warnings), 1)
        self.assertIn("predictor coverage 0.0%", warnings[0])
        self.assertIn("still to be played", warnings[0])

    def test_mixed_slate_measures_only_the_unplayed_games(self) -> None:
        games = [_game(predictor=False, final=True) for _ in range(10)]
        games += [_game(predictor=True) for _ in range(4)]
        games += [_game(predictor=False)]
        summary = summarize_coverage(games)
        self.assertEqual(summary["predictorEligible"], 5)
        self.assertEqual(summary["predictorPresent"], 4)
        self.assertEqual(summary["predictorPct"], 80.0)
        self.assertEqual(_predictor_warnings(summary), [])

    def test_voided_games_are_not_eligible(self) -> None:
        summary = summarize_coverage([_game(predictor=False, voided=True) for _ in range(5)])
        self.assertEqual(summary["predictorEligible"], 0)

    def test_slate_wide_counts_are_unchanged(self) -> None:
        """The dashboard's coverage display still means 'of this slate'."""
        games = [_game(predictor=True) for _ in range(2)]
        games += [_game(predictor=False, final=True) for _ in range(2)]
        summary = summarize_coverage(games)
        self.assertEqual(summary["gameCount"], 4)
        self.assertEqual(summary["counts"]["espnPredictor"], 2)
        self.assertEqual(summary["pct"]["espnPredictor"], 50.0)


if __name__ == "__main__":
    unittest.main()
