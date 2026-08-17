"""One NaN feature did not degrade the fit. It destroyed it.

Found 17 Aug 2026 by fuzzing `fit_logistic` with degenerate inputs. IRLS
accumulates each row into the gradient and the Hessian, and a non-finite value
propagates through both, so it is not the offending row that is lost -- it is
every coefficient. Measured before the fix, ten NaN rows among twenty:

    fit_logistic([[1,nan]]*10 + [[1,1]]*10, [0]*10 + [1]*10)  ->  [nan, nan]

NaN weights make every prediction NaN, every published probability NaN, and --
before `write_json` -- every page that reads them unparseable. The fitter also
survived a `nan` reaching it silently, because `float("nan")` converts without
complaining and `_first_number` returned it.

The same fuzz found `zip(rows, labels)` truncating to the shorter sequence, so
a caller that lost the row/outcome correspondence would train on a prefix while
reporting the full count:

    fit_logistic([[1,1],[1,2]], [1])  ->  [21.995, 0.0]   (one row used)

Latent rather than live: all 23,525 float feature values in the current
`predictions_log.json` are finite, and refitting the real log with and without
this change produces a bit-identical `model_weights.json`. What changed is what
happens the first time a provider emits a division by zero.

Two guards, on the same principle as `_row_units`: a bad value costs its own
row, not the whole run. `_first_number` treats a non-finite number as absent --
a shape the pipeline already handles -- and `fit_logistic` drops any row that
still carries one, as a backstop for rows built by some other path.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_fit import _first_number, fit_logistic

NON_FINITE = (float("nan"), float("inf"), float("-inf"))


class FirstNumberTests(unittest.TestCase):
    def test_a_non_finite_value_reads_as_absent(self) -> None:
        for value in NON_FINITE:
            self.assertIsNone(_first_number(value))

    def test_it_falls_through_to_the_next_usable_value(self) -> None:
        """Same contract as None: this source had nothing, try the next."""
        self.assertEqual(_first_number(float("nan"), 3), 3.0)
        self.assertEqual(_first_number(None, float("inf"), 2.5), 2.5)

    def test_real_numbers_are_unchanged(self) -> None:
        self.assertEqual(_first_number(2.5), 2.5)
        self.assertEqual(_first_number("2.5"), 2.5)
        self.assertEqual(_first_number(0), 0.0)
        self.assertEqual(_first_number(-1), -1.0)

    def test_zero_is_a_value_and_not_a_miss(self) -> None:
        """`or`-style fallbacks get this wrong; this one must not."""
        self.assertEqual(_first_number(0, 5), 0.0)

    def test_nothing_usable_is_none(self) -> None:
        self.assertIsNone(_first_number())
        self.assertIsNone(_first_number(None, "abc", [], float("nan")))


def _separable(n: int = 200) -> tuple[list[list[float]], list[int]]:
    rows = [[1.0, 1.0 if i % 2 else -1.0] for i in range(n)]
    labels = [1 if i % 2 else 0 for i in range(n)]
    return rows, labels


class PoisonedRowTests(unittest.TestCase):
    def test_a_fit_never_returns_non_finite_weights(self) -> None:
        for value in NON_FINITE:
            with self.subTest(value=value):
                rows = [[1.0, value]] * 10 + [[1.0, 1.0]] * 10
                weights = fit_logistic(rows, [0] * 10 + [1] * 10)
                self.assertTrue(all(math.isfinite(w) for w in weights), weights)

    def test_one_bad_row_costs_one_row_and_no_more(self) -> None:
        """The remaining fit must be the fit of the remaining data."""
        rows, labels = _separable()
        clean = fit_logistic(rows, labels)
        poisoned = fit_logistic(rows + [[1.0, float("nan")]], labels + [1])
        self.assertEqual(len(clean), len(poisoned))
        for before, after in zip(clean, poisoned):
            self.assertAlmostEqual(before, after, places=1)

    def test_a_fit_with_nothing_usable_left_returns_nothing(self) -> None:
        """Empty is the honest answer; zeros would be a model that predicts."""
        self.assertEqual(fit_logistic([[1.0, float("nan")]] * 10, [0] * 10), [])

    def test_a_row_whose_label_is_missing_is_dropped(self) -> None:
        rows, labels = _separable(20)
        weights = fit_logistic(rows + [[1.0, 1.0]], labels + [None])
        self.assertTrue(all(math.isfinite(w) for w in weights))


class LengthMismatchTests(unittest.TestCase):
    def test_more_rows_than_labels_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_logistic([[1.0, 1.0], [1.0, 2.0]], [1])

    def test_more_labels_than_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_logistic([[1.0, 1.0]], [1, 0])

    def test_the_message_names_both_counts(self) -> None:
        with self.assertRaises(ValueError) as caught:
            fit_logistic([[1.0, 1.0], [1.0, 2.0]], [1])
        self.assertIn("2", str(caught.exception))
        self.assertIn("1", str(caught.exception))

    def test_matched_lengths_are_untouched(self) -> None:
        rows, labels = _separable(20)
        self.assertEqual(len(fit_logistic(rows, labels)), 2)


class UnchangedBehaviourTests(unittest.TestCase):
    """Everything the fitter already did, it must still do."""

    def test_empty_input_is_still_empty_output(self) -> None:
        self.assertEqual(fit_logistic([], []), [])

    def test_a_separable_problem_still_fits(self) -> None:
        rows, labels = _separable()
        weights = fit_logistic(rows, labels)
        self.assertGreater(weights[1], 1.0, "the discriminating feature earned no weight")

    def test_regularisation_still_bounds_a_separable_fit(self) -> None:
        """Without a penalty the coefficient runs to the iteration limit; with
        one it must not."""
        rows, labels = _separable()
        self.assertLess(fit_logistic(rows, labels, l2=1.0)[1], fit_logistic(rows, labels, l2=0.0)[1])

    def test_a_per_feature_penalty_of_the_wrong_width_still_raises(self) -> None:
        rows, labels = _separable(20)
        with self.assertRaises(ValueError):
            fit_logistic(rows, labels, l2=[1.0, 1.0, 1.0])

    def test_collinear_features_do_not_break_the_solve(self) -> None:
        rows = [[1.0, float(i % 7), float(i % 7)] for i in range(60)]
        labels = [1 if i % 7 > 3 else 0 for i in range(60)]
        weights = fit_logistic(rows, labels)
        self.assertTrue(all(math.isfinite(w) for w in weights), weights)


if __name__ == "__main__":
    unittest.main()
