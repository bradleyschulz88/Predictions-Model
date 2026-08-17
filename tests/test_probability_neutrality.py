"""A probability that is not a number became a 100% pick.

Found 17 Aug 2026 probing the calibration boundary. `logit` clamped with

    prob = max(1e-6, min(1.0 - 1e-6, prob))

and every comparison against NaN is False, so `min(0.999999, nan)` returns
0.999999. The clamp came out at its CEILING:

    logit(nan)              -> 13.815510
    apply_platt(nan, ...)   -> 0.999999

A NaN probability therefore published as a 100.0% pick -- and confidence drives
the Kelly stake, so the least trustworthy input in the system produced the
largest bet in it. Which end it landed on was decided by nothing more than the
argument order: `max(1e-6, nan)` returns 1e-06, so writing the same clamp the
other way round would have sent it to the floor instead.

Neither end is right. An input that means "no idea" must read as 0.5 -- below
every publication threshold, and a stake of nothing.

`sigmoid` had the opposite half of the problem: it propagated NaN faithfully
(`math.exp(nan)` is NaN) and so passed the poison on rather than absorbing it.
Infinity is different and is left alone: an infinite logit is a real saturation
and maps to the end it points at.

Nothing produced a NaN probability at the time of writing -- the fitter and the
feature builder were both hardened in the same review -- so this is the third
guard on the same path rather than a live fault. It is the one that matters
most, because it is the last place a bad number can be caught before it becomes
a published bet.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from calibration_params import apply_platt, min_pick_confidence
from model_fit import NEUTRAL_PROBABILITY, logit, predict_row, sigmoid

NAN = float("nan")


class LogitTests(unittest.TestCase):
    def test_a_nan_probability_reads_as_no_opinion(self) -> None:
        self.assertEqual(logit(NAN), 0.0)

    def test_it_no_longer_lands_on_the_clamp_ceiling(self) -> None:
        """The specific defect: 13.815510, one clamp step from certainty."""
        self.assertNotAlmostEqual(logit(NAN), logit(1.0), places=6)

    def test_an_infinity_is_also_treated_as_absent(self) -> None:
        """An infinite probability is not a saturation, it is a broken number."""
        self.assertEqual(logit(float("inf")), 0.0)
        self.assertEqual(logit(float("-inf")), 0.0)

    def test_something_that_is_not_a_number_at_all_is_absent_too(self) -> None:
        self.assertEqual(logit(None), 0.0)
        self.assertEqual(logit("nonsense"), 0.0)

    def test_the_ordinary_range_is_completely_unchanged(self) -> None:
        for prob, expected in ((0.5, 0.0), (0.75, 1.0986122886681098), (0.25, -1.0986122886681098)):
            self.assertAlmostEqual(logit(prob), expected, places=9)

    def test_the_ends_still_clamp_rather_than_diverge(self) -> None:
        self.assertAlmostEqual(logit(0.0), -13.815509557963773, places=6)
        self.assertAlmostEqual(logit(1.0), 13.815509557935018, places=6)

    def test_out_of_range_input_still_clamps_to_the_near_end(self) -> None:
        """Unlike NaN, -0.5 has a direction, so it keeps it."""
        self.assertEqual(logit(-0.5), logit(0.0))
        self.assertEqual(logit(1.5), logit(1.0))


class SigmoidTests(unittest.TestCase):
    def test_a_nan_logit_reads_as_no_opinion(self) -> None:
        self.assertEqual(sigmoid(NAN), NEUTRAL_PROBABILITY)

    def test_an_infinite_logit_still_saturates_in_its_own_direction(self) -> None:
        """This one is a real answer, not a missing one."""
        self.assertEqual(sigmoid(float("inf")), 1.0)
        self.assertEqual(sigmoid(float("-inf")), 0.0)

    def test_the_ordinary_range_is_completely_unchanged(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5, places=12)
        self.assertAlmostEqual(sigmoid(2.0), 0.8807970779778823, places=12)
        self.assertAlmostEqual(sigmoid(-2.0), 0.11920292202211755, places=12)

    def test_large_magnitudes_do_not_overflow(self) -> None:
        self.assertEqual(sigmoid(1000.0), 1.0)
        self.assertEqual(sigmoid(-1000.0), 0.0)

    def test_the_round_trip_survives(self) -> None:
        for prob in (0.01, 0.25, 0.5, 0.62, 0.9, 0.999):
            self.assertAlmostEqual(sigmoid(logit(prob)), prob, places=9)


class CalibrationTests(unittest.TestCase):
    """The published consequence."""

    def test_a_nan_no_longer_calibrates_to_near_certainty(self) -> None:
        self.assertAlmostEqual(apply_platt(NAN, {"a": 1.0, "b": 0.0}), 0.5, places=9)

    def test_the_result_is_below_every_publication_threshold(self) -> None:
        """The point of choosing the middle: it publishes nothing and stakes
        nothing, rather than the maximum of both."""
        calibrated = apply_platt(NAN, {"a": 1.0, "b": 0.0}) * 100
        for league in ("mlb", "nfl", "nba", "wnba", "epl", "afl"):
            self.assertLess(calibrated, min_pick_confidence(league), league)

    def test_a_real_probability_passes_through_an_identity_curve(self) -> None:
        for prob in (0.35, 0.5, 0.62, 0.88):
            self.assertAlmostEqual(apply_platt(prob, {"a": 1.0, "b": 0.0}), prob, places=9)

    def test_a_fitted_curve_still_moves_the_number(self) -> None:
        shrunk = apply_platt(0.8, {"a": 0.5, "b": 0.0})
        self.assertLess(shrunk, 0.8)
        self.assertGreater(shrunk, 0.5)


class PredictionPathTests(unittest.TestCase):
    """The route a bad feature would take to the board."""

    def test_a_nan_feature_at_prediction_time_yields_no_opinion(self) -> None:
        """Weights are fine, the feature is not -- so the fitter's guard, which
        runs at fit time, cannot help here."""
        self.assertEqual(predict_row([0.1, 0.5], [1.0, NAN]), NEUTRAL_PROBABILITY)

    def test_a_nan_weight_yields_no_opinion(self) -> None:
        self.assertEqual(predict_row([NAN, 0.5], [1.0, 1.0]), NEUTRAL_PROBABILITY)

    def test_a_clean_row_predicts_normally(self) -> None:
        predicted = predict_row([0.1, 0.5], [1.0, 1.0])
        self.assertTrue(math.isfinite(predicted))
        self.assertAlmostEqual(predicted, sigmoid(0.6), places=12)


if __name__ == "__main__":
    unittest.main()
