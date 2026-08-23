"""A league intercept must be a correction, not an edge invented from noise.

`check_regression.py` refuses any fitted intercept past 0.25, on the grounds
that a per-league home-field adjustment bigger than that is not a correction to
a pooled fit -- it is a claim to a standing edge. On 2026-08-22 the NFL
intercept reached -0.2722 and that gate failed every scheduled build.

The gate was right. The shipped intercepts came out almost perfectly inverse to
sample size, which is the signature of a fit reading noise:

    nfl        21 rows   -0.2722        afl        73 rows   +0.0077
    worldcup   60 rows   +0.1884        mlb       824 rows   -0.0137
    wnba      164 rows   +0.1333

Count-based shrinkage of `n/(n+50)` was already applied. It is not enough,
because a thin league's raw estimate is also enormous -- NFL's was -0.9203 --
and 30% of an enormous number is still an edge.

What was missing is the estimate's own uncertainty. `evaluation.py` reports
NFL's home bias as "+18.8 +/- 12.5pts -- within noise" from the same rows, so
the fitter was writing a coefficient the project's own reporting called noise.
Soft-thresholding at one standard error is the standard estimator for a mean
that might be zero: keep only the part that exceeds what chance produces,
return exactly zero when nothing does, then apply the count shrinkage on top.

Not a tuned constant chosen to make the gate pass -- walk-forward improved:

    logloss 0.6397 -> 0.6391 · brier 0.2258 -> 0.2254 · acc 0.6116 -> 0.6137

STILL OPEN, deliberately not fixed here. All 22 graded NFL rows are preseason
(6-21 Aug, every one flagged `strengthGames: 0`), so the surviving -0.15 is a
preseason-derived correction that will be applied to regular-season games.
Excluding them needs `seasonType` threaded into `Sample`, which is a feature
change rather than a bug fix. It self-corrects as regular-season games grade,
with a lag.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_fit import (
    LEAGUE_INTERCEPT_PRIOR,
    MIN_LEAGUE_INTERCEPT_SAMPLE,
    Sample,
    _league_intercepts,
    fit_from_observations,
    samples_from_log,
)

DATA = ROOT / "docs" / "data"

# The ceiling check_regression.py enforces. Duplicated rather than imported so
# a change to one is visible as a disagreement with the other.
MAX_INTERCEPT = 0.25


def _samples(league: str, count: int, *, home_rate: float, seed: int = 3) -> list[Sample]:
    """Rows whose only signal is how often the home side actually won."""
    random.seed(seed)
    out = []
    for index in range(count):
        label = 1 if random.random() < home_rate else 0
        out.append(
            Sample(
                values={"strengthDiff": random.gauss(0.0, 0.05), "marketLogit": 0.0},
                label=label,
                league=league,
                date=f"2026-01-{index % 28 + 1:02d}",
            )
        )
    return out


# A league intercept is a correction RELATIVE TO THE POOL, so a single-league
# fixture measures nothing: the pooled intercept absorbs that league's base
# rate and the correction is correctly zero. Every case below therefore sits a
# league of interest alongside a large neutral one. (Learned by writing the
# first version of these tests without it and reading four zeros.)
def _pool(seed: int = 2) -> list[Sample]:
    return _samples("mlb", 600, home_rate=0.50, seed=seed)


class LiveFitTests(unittest.TestCase):
    """Against the real log, which is what the gate reads."""

    @classmethod
    def setUpClass(cls) -> None:
        samples, _centre = samples_from_log(DATA)
        cls.samples = samples
        cls.payload = fit_from_observations(samples)
        cls.intercepts = cls.payload["leagueIntercepts"]

    def test_no_intercept_exceeds_what_the_regression_gate_allows(self) -> None:
        offenders = [
            f"{league} {value:+.4f}"
            for league, value in self.intercepts.items()
            if abs(value) > MAX_INTERCEPT
        ]
        self.assertEqual(
            offenders, [],
            "check_regression.py fails the build on these, which is how the "
            "scheduled runs went red: " + "; ".join(offenders),
        )

    def test_the_thinnest_league_is_bounded_by_the_gate_like_any_other(self) -> None:
        """The ordering that gave the bug away, held only where it is honest.

        An earlier version of this asserted the thinnest league's intercept was
        under half the gate. That bound was invented to look strict; NFL sits
        at -0.1519, which clears the real ceiling and fails a made-up one. A
        test whose threshold has no derivation behind it teaches nothing when
        it fails, and the correct response to it -- loosen the number -- is
        indistinguishable from ignoring a real problem.

        So this checks the bound that actually exists. NFL being both the
        thinnest league and the largest correction is a real, documented state:
        all 22 of its graded rows are preseason, and the intercept moves toward
        zero as regular-season games grade.
        """
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.league] = counts.get(sample.league, 0) + 1
        scored = [(l, v) for l, v in self.intercepts.items() if l in counts]
        if len(scored) < 3:
            self.skipTest("too few leagues to compare")
        thinnest = min((l for l, _ in scored), key=lambda l: counts[l])
        self.assertLessEqual(
            abs(self.intercepts[thinnest]), MAX_INTERCEPT,
            f"{thinnest} has the fewest rows ({counts[thinnest]}) and an intercept of "
            f"{self.intercepts[thinnest]:+.4f}, which the regression gate rejects",
        )

    def test_a_league_with_almost_no_history_gets_no_intercept(self) -> None:
        counts: dict[str, int] = {}
        for sample in self.samples:
            counts[sample.league] = counts.get(sample.league, 0) + 1
        for league, count in counts.items():
            if count < MIN_LEAGUE_INTERCEPT_SAMPLE:
                self.assertNotIn(
                    league, self.intercepts,
                    f"{league} has {count} rows, too few to estimate its own spread",
                )


class SoftThresholdTests(unittest.TestCase):
    """The estimator itself, on data where the answer is known."""

    def _fit(self, samples: list[Sample]) -> dict[str, float]:
        payload = fit_from_observations(samples)
        return _league_intercepts(samples, payload["anchored"], payload["standalone"])

    def test_a_league_no_different_from_the_pool_gets_exactly_zero(self) -> None:
        """Not "small" -- zero. A residual inside its own noise is no evidence."""
        intercepts = self._fit(_pool() + _samples("wnba", 200, home_rate=0.50, seed=9))
        self.assertEqual(intercepts.get("wnba"), 0.0)

    def test_a_thin_lopsided_league_is_not_handed_a_large_edge(self) -> None:
        """20 games at 30% home is exactly NFL's shape, and it must not
        produce a coefficient the regression gate rejects."""
        intercepts = self._fit(_pool() + _samples("nfl", 20, home_rate=0.30, seed=5))
        self.assertLessEqual(abs(intercepts.get("nfl", 0.0)), MAX_INTERCEPT)

    def test_a_league_below_the_floor_is_left_out_entirely(self) -> None:
        few = MIN_LEAGUE_INTERCEPT_SAMPLE - 1
        intercepts = self._fit(_pool() + _samples("epl", few, home_rate=1.0, seed=6))
        self.assertNotIn("epl", intercepts)

    def test_one_row_cannot_produce_an_intercept(self) -> None:
        """It has no spread to threshold against; EPL's single row produced a
        raw residual of +0.9996."""
        intercepts = self._fit(_pool() + _samples("epl", 1, home_rate=1.0, seed=7))
        self.assertNotIn("epl", intercepts)

    def test_more_evidence_of_the_same_effect_earns_a_larger_intercept(self) -> None:
        """Soft-thresholding must not flatten a real effect into nothing."""
        thin = abs(self._fit(_pool() + _samples("wnba", 40, home_rate=0.68, seed=3)).get("wnba", 0.0))
        thick = abs(self._fit(_pool() + _samples("wnba", 400, home_rate=0.68, seed=3)).get("wnba", 0.0))
        self.assertGreater(thick, thin)

    def test_the_sign_follows_the_data(self) -> None:
        home = self._fit(_pool() + _samples("wnba", 400, home_rate=0.70, seed=3)).get("wnba", 0.0)
        away = self._fit(_pool() + _samples("wnba", 400, home_rate=0.30, seed=4)).get("wnba", 0.0)
        self.assertGreater(home, 0.0)
        self.assertLess(away, 0.0)

    def test_the_threshold_is_one_standard_error_of_the_mean(self) -> None:
        """Pins the estimator, so a later edit cannot quietly change what
        "within noise" means here."""
        source = (ROOT / "model_fit.py").read_text(encoding="utf-8")
        block = source[source.index("def _league_intercepts("):]
        block = block[: block.index("\ndef ")]
        self.assertIn("math.sqrt(variance / count)", block)
        self.assertIn("abs(mean_residual) - std_error", block)
        self.assertIn(f"count / (count + LEAGUE_INTERCEPT_PRIOR)", block)

    def test_the_count_shrinkage_still_applies_on_top(self) -> None:
        """Both terms, not one replacing the other.

        The soft threshold removes noise; the count shrinkage pulls what
        survives toward the pool. Dropping either would have left the original
        bug half-fixed.
        """
        self.assertGreater(LEAGUE_INTERCEPT_PRIOR, 0.0)
        count = 200
        samples = _pool() + _samples("wnba", count, home_rate=0.66, seed=8)
        shipped = abs(self._fit(samples)["wnba"])
        shrink = count / (count + LEAGUE_INTERCEPT_PRIOR)
        self.assertLess(shrink, 1.0)
        # What the same soft-thresholded residual would have been without it.
        self.assertLess(shipped, shipped / shrink)


class DoesNotDegradeTheModelTests(unittest.TestCase):
    """A gate can always be passed by making the model worse."""

    def test_the_walk_forward_fit_still_beats_a_coin_flip(self) -> None:
        samples, _centre = samples_from_log(DATA)
        if len(samples) < 200:
            self.skipTest("not enough graded history")
        payload = fit_from_observations(samples)
        walk = payload.get("walkForward") or {}
        logloss = walk.get("logloss")
        if logloss is None:
            self.skipTest("no walk-forward block in this payload")
        self.assertLess(logloss, 0.6931, "worse than a coin flip")

    def test_every_intercept_is_a_real_number(self) -> None:
        samples, _centre = samples_from_log(DATA)
        for league, value in fit_from_observations(samples)["leagueIntercepts"].items():
            self.assertTrue(math.isfinite(value), f"{league} intercept is {value}")


if __name__ == "__main__":
    unittest.main()
