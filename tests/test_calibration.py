"""Tests for Platt calibration and the feedback loop it replaces."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import calibration_params as cal  # noqa: E402
from calibration_params import (  # noqa: E402
    apply_platt,
    compute_platt_params,
    fit_platt,
    is_publishable_pick,
    platt_params_for,
)
from model_fit import sigmoid  # noqa: E402


def _overconfident_pairs(count: int = 600, seed: int = 4) -> list[tuple[float, int]]:
    """Probabilities that are systematically too extreme."""
    random.seed(seed)
    pairs = []
    for _ in range(count):
        true_prob = random.uniform(0.35, 0.65)
        # Stated probability pushed away from 0.5 -- classic overconfidence.
        stated = min(0.99, max(0.01, 0.5 + (true_prob - 0.5) * 2.5))
        pairs.append((stated, 1 if random.random() < true_prob else 0))
    return pairs


class PlattFitTests(unittest.TestCase):
    def test_identity_when_sample_is_too_small(self) -> None:
        params = fit_platt([(0.7, 1), (0.6, 0)])
        self.assertAlmostEqual(params["a"], 1.0)
        self.assertAlmostEqual(params["b"], 0.0)

    def test_pulls_overconfident_probabilities_toward_the_middle(self) -> None:
        params = fit_platt(_overconfident_pairs())
        # A slope below 1 flattens the curve, which is the correction needed.
        self.assertLess(params["a"], 1.0)
        corrected = apply_platt(0.90, params)
        self.assertLess(corrected, 0.90)
        self.assertGreater(corrected, 0.5)

    def test_can_express_worse_than_a_coin_flip(self) -> None:
        """Bucket shrinkage clamped at 0.5 and lost this; Platt can say it."""
        random.seed(9)
        # Stated probability is anti-correlated with the outcome.
        pairs = [
            (0.8 if outcome == 0 else 0.2, outcome)
            for outcome in (random.randint(0, 1) for _ in range(400))
        ]
        params = fit_platt(pairs)
        self.assertLess(params["a"], 0.0)

    def test_thin_data_stays_closer_to_the_identity_curve(self) -> None:
        small = fit_platt(_overconfident_pairs(count=60))
        large = fit_platt(_overconfident_pairs(count=2000))
        # More evidence earns more correction; a short history only nudges.
        self.assertLess(abs(small["a"] - 1.0), abs(large["a"] - 1.0))

    def test_identity_params_leave_probabilities_untouched(self) -> None:
        for prob in (0.05, 0.3, 0.5, 0.77, 0.95):
            self.assertAlmostEqual(apply_platt(prob, dict(cal.IDENTITY_PLATT)), prob, places=6)

    def test_zero_intercept_curve_is_symmetric_about_a_half(self) -> None:
        """With no base-rate correction, the curve treats both sides alike."""
        params = {"a": 0.6, "b": 0.0, "n": 500}
        self.assertAlmostEqual(apply_platt(0.8, params) + apply_platt(0.2, params), 1.0, places=6)

    def test_intercept_shifts_both_sides_the_same_way(self) -> None:
        """A non-zero intercept is a deliberate base-rate correction.

        This is what fixes a model that picks home more often than home wins, so
        it must move low probabilities up as well as high ones.
        """
        params = {"a": 1.0, "b": -0.4, "n": 500}
        self.assertLess(apply_platt(0.8, params), 0.8)
        self.assertLess(apply_platt(0.2, params), 0.2)

    def test_curve_is_monotonic_across_the_whole_range(self) -> None:
        params = fit_platt(_overconfident_pairs())
        values = [apply_platt(prob / 100.0, params) for prob in range(5, 96)]
        self.assertEqual(values, sorted(values))

    def test_missing_params_are_a_no_op(self) -> None:
        self.assertAlmostEqual(apply_platt(0.73, None), 0.73)


class PlattLookupTests(unittest.TestCase):
    def _calibration(self) -> dict:
        return {
            "plattParams": {
                "fitted": {
                    "mlb": {"a": 0.8, "b": 0.1, "n": 200},
                    "default": {"a": 0.9, "b": 0.0, "n": 500},
                    "afl": {"a": 0.5, "b": 0.0, "n": 5},
                },
                "heuristic": {"default": {"a": 0.4, "b": 0.0, "n": 300}},
            }
        }

    def test_prefers_the_league_curve(self) -> None:
        params = platt_params_for(self._calibration(), method="fitted", league="mlb")
        self.assertAlmostEqual(params["a"], 0.8)

    def test_falls_back_to_pooled_when_league_is_thin(self) -> None:
        params = platt_params_for(self._calibration(), method="fitted", league="afl")
        self.assertAlmostEqual(params["a"], 0.9)

    def test_falls_back_to_identity_for_an_unknown_method(self) -> None:
        params = platt_params_for(self._calibration(), method="xgboost", league="mlb")
        self.assertAlmostEqual(params["a"], 1.0)

    def test_methods_do_not_share_curves(self) -> None:
        """A curve fitted on the heuristic's errors must not reach the fitted model."""
        fitted = platt_params_for(self._calibration(), method="fitted", league="nba")
        heuristic = platt_params_for(self._calibration(), method="heuristic", league="nba")
        self.assertNotAlmostEqual(fitted["a"], heuristic["a"])


class FeedbackLoopTests(unittest.TestCase):
    """The old calibrator learned from its own previous corrections."""

    def test_records_without_raw_probability_are_skipped(self) -> None:
        graded = [
            {"confidence": 80.0, "correct": True, "homeScore": 5, "awayScore": 3, "league": "mlb"}
            for _ in range(200)
        ]
        # Published confidence exists but rawHomeWinPct does not, so nothing is
        # fitted -- rather than silently recalibrating on calibrated values.
        self.assertEqual(compute_platt_params(graded), {})

    def test_fits_from_raw_home_probability(self) -> None:
        random.seed(2)
        graded = []
        for _ in range(300):
            raw = random.uniform(0.2, 0.8)
            home_won = random.random() < raw
            graded.append(
                {
                    "rawHomeWinPct": raw * 100,
                    "probabilityMethod": "fitted",
                    "league": "mlb",
                    "homeScore": 5 if home_won else 2,
                    "awayScore": 2 if home_won else 5,
                }
            )
        params = compute_platt_params(graded)
        self.assertIn("fitted", params)
        self.assertGreaterEqual(params["fitted"]["mlb"]["n"], 200)

    def test_draws_are_excluded_from_the_binary_fit(self) -> None:
        graded = [
            {
                "rawHomeWinPct": 60.0,
                "probabilityMethod": "fitted",
                "league": "epl",
                "homeScore": 1,
                "awayScore": 1,
            }
            for _ in range(100)
        ]
        self.assertEqual(compute_platt_params(graded), {})

    def test_a_well_calibrated_model_gets_a_near_identity_curve(self) -> None:
        random.seed(6)
        graded = []
        for _ in range(800):
            raw = sigmoid(random.gauss(0.0, 0.8))
            home_won = random.random() < raw
            graded.append(
                {
                    "rawHomeWinPct": raw * 100,
                    "probabilityMethod": "fitted",
                    "league": "mlb",
                    "homeScore": 5 if home_won else 2,
                    "awayScore": 2 if home_won else 5,
                }
            )
        curve = compute_platt_params(graded)["fitted"]["mlb"]
        self.assertAlmostEqual(curve["a"], 1.0, delta=0.25)
        self.assertAlmostEqual(curve["b"], 0.0, delta=0.25)


class PublishThresholdTests(unittest.TestCase):
    def _app_js(self) -> str:
        return (ROOT / "dashboard" / "app.js").read_text(encoding="utf-8")

    def test_dashboard_fallback_matches_the_python_threshold(self) -> None:
        """The fallback is what paints before calibration.json loads."""
        expected = f"const MIN_PUBLISHABLE_CONFIDENCE = {cal.MIN_PICK_CONFIDENCE};"
        self.assertIn(expected, self._app_js())

    def test_dashboard_reads_the_live_threshold_from_data(self) -> None:
        app_js = self._app_js()
        self.assertIn("minPickConfidence", app_js)
        self.assertIn("function minPublishableConfidence()", app_js)
        # The publish check must consult the data, not the baked-in constant.
        self.assertIn("return confidence >= minPublishableConfidence();", app_js)

    def test_dashboard_reads_tier_thresholds_from_data(self) -> None:
        """Tier boundaries were duplicated in JS and drifted from the model."""
        app_js = self._app_js()
        self.assertIn("function confidenceTiers()", app_js)
        self.assertIn("calibrationData?.thresholds", app_js)
        self.assertIn("value >= tiers.strong", app_js)
        self.assertIn("value >= tiers.lean", app_js)

    def test_no_bare_tier_literals_remain_in_the_label_logic(self) -> None:
        app_js = self._app_js()
        for literal in ("confidence >= 68", "confidence >= 57", "value >= 68", "value >= 57"):
            self.assertNotIn(literal, app_js, msg=f"hardcoded tier threshold: {literal}")

    def test_backtest_publishes_the_thresholds_the_dashboard_reads(self) -> None:
        from scripts.backtest_model import LEAN_THRESHOLD, STRONG_THRESHOLD

        report = {"summary": {"graded": 0}, "calibration": [], "calibrationByLeague": {}}
        params = cal.compute_calibration_params(report)
        self.assertEqual(params["minPickConfidence"], cal.MIN_PICK_CONFIDENCE)
        # The tier boundaries the UI renders come from this same report.
        self.assertEqual(STRONG_THRESHOLD, cal.STRONG_THRESHOLD)
        self.assertEqual(LEAN_THRESHOLD, cal.LEAN_THRESHOLD)

    def test_pick_below_threshold_is_not_published(self) -> None:
        self.assertFalse(
            is_publishable_pick({"predictedWinner": "A", "confidence": cal.MIN_PICK_CONFIDENCE - 0.1})
        )

    def test_pick_at_threshold_is_published(self) -> None:
        self.assertTrue(
            is_publishable_pick({"predictedWinner": "A", "confidence": cal.MIN_PICK_CONFIDENCE})
        )

    def test_missing_confidence_is_not_published(self) -> None:
        self.assertFalse(is_publishable_pick({"predictedWinner": "A"}))
        self.assertFalse(is_publishable_pick(None))


if __name__ == "__main__":
    unittest.main()
