"""The residual estimator must reproduce a value already known to be right.

MARGIN_STD_DEV needs the spread of a margin around one game's expected margin.
wnba 13.49 and afl 40.05 hold the spread across all games instead, which is
larger, and measuring raw margins harder only makes that worse -- NBA measures
16.21 across 1059 games where the right answer is about 11.5.

The estimator uses the decomposition rather than more measurement:

    Var(margin over all games) = Var(expected margin) + Var(residual)
    sigma_r = SD(margin) / sqrt(1 + Var(z)),  z = inverse-normal of market prob
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.estimate_margin_residual import MIN_PRICED_GAMES, residual_sd  # noqa: E402


def _z_scores(n: int, spread: float) -> list[float]:
    """n inverse-normal scores with a target standard deviation."""
    step = 2.0 / (n - 1)
    raw = [(-1.0 + i * step) for i in range(n)]
    scale = spread / math.sqrt(sum(v * v for v in raw) / n)
    return [v * scale for v in raw]


class ResidualEstimatorTests(unittest.TestCase):
    def test_reproduces_the_nba_value_known_to_be_correct(self) -> None:
        """16.21 raw with NBA-shaped market spread must give about 11.5.

        This is the check that the method is sound: 11.5 is the entry the
        spread-cover test already validates, arrived at independently.
        """
        z = _z_scores(400, math.sqrt(0.99))
        sigma = residual_sd(16.21, z)
        self.assertAlmostEqual(sigma, 11.5, delta=0.2)

    def test_a_market_that_prices_every_game_even_returns_the_raw_sd(self) -> None:
        """With no strength variation there is no second term to remove."""
        sigma = residual_sd(15.0, [0.0] * 400)
        self.assertAlmostEqual(sigma, 15.0, places=6)

    def test_a_wider_market_spread_implies_a_smaller_residual(self) -> None:
        narrow = residual_sd(15.0, _z_scores(400, 0.4))
        wide = residual_sd(15.0, _z_scores(400, 1.2))
        self.assertLess(wide, narrow)
        self.assertLess(wide, 15.0)

    def test_the_residual_is_never_larger_than_the_raw_sd(self) -> None:
        """Variance decomposes; the part cannot exceed the whole."""
        for spread in (0.0, 0.3, 0.8, 1.5, 3.0):
            self.assertLessEqual(residual_sd(15.0, _z_scores(400, spread)), 15.0 + 1e-9)

    def test_a_thin_sample_refuses_to_answer(self) -> None:
        """A plausible number measured on nothing is what caused this bug."""
        self.assertIsNone(residual_sd(15.16, _z_scores(MIN_PRICED_GAMES - 1, 0.9)))

    def test_it_answers_once_the_sample_is_large_enough(self) -> None:
        self.assertIsNotNone(residual_sd(15.16, _z_scores(MIN_PRICED_GAMES, 0.9)))

    def test_the_current_wnba_entry_is_larger_than_any_residual_it_could_have(self) -> None:
        """13.49 cannot be a residual: it exceeds what the decomposition allows.

        WNBA's raw margin SD is 15.16. A residual of 13.49 would mean expected
        margins vary by only sqrt(15.16^2 - 13.49^2) = 6.9 points across the
        league, which no basketball market looks like -- confirming the entry
        holds the across-all-games figure.
        """
        implied_strength_spread = math.sqrt(15.16**2 - 13.49**2)
        self.assertLess(implied_strength_spread, 7.0)
        normal = NormalDist()
        # A 6.9-point spread of expected margins puts a one-sigma favourite at
        # only this win probability, which is far too flat for the sport.
        self.assertLess(normal.cdf(implied_strength_spread / 13.49), 0.71)


if __name__ == "__main__":
    unittest.main()
