"""Divergence from the market, reported for the model that is actually running.

Pooled over every graded pick, this said the model strays from the market by a
19.2-point median with 57.5% of games over 15 points, and that 116 picks fade
the price at 52.6% against a 52.4% break-even -- a board's worth of coin flips.

Measured 2026-08-05, split by era:

    era                  n    median gap   >15pt   fade share
    2026-06-18..07-01  209      14.0pts    45.9%      23.0%
    2026-07-01..07-21  209      12.4pts    36.4%      22.5%
    2026-07-21..08-04  210       3.9pts     6.7%      12.4%

The problem was real and was fixed weeks ago. What the pooled figure describes
is a forecaster that no longer exists, and a refit changes what the model says,
so pooling across versions is the same error reliability already avoids.

Shrinking further toward the market would now make things worse, which is the
action the pooled number invites. On the most recent 200 games, out-of-sample
log loss rises monotonically with the market weight: 0.6369 at w=0, 0.6385 at
0.4, 0.6479 at w=1. The optimum on the pooled record is w=0.58; on current data
it is near zero.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.evaluation import Observation, divergence_report  # noqa: E402


def _obs(model: float, market: float, home_won: int, *, current: bool, index: int = 0):
    return Observation(
        event_id=f"e{index}", league="mlb", date="2026-07-01", home_won=home_won,
        model=model, market=market, published=model, current_pipeline=current,
    )


def _old_wide(count: int, index: int = 0):
    """Old-pipeline picks that diverge hugely, half of them wrong."""
    return [
        _obs(0.85, 0.45, i % 2, current=False, index=index + i)
        for i in range(count)
    ]


def _new_tight(count: int, index: int = 0):
    """Current-pipeline picks that sit close to the price."""
    return [
        _obs(0.56, 0.54, 1, current=True, index=index + i)
        for i in range(count)
    ]


class CurrentPipelineSplitTests(unittest.TestCase):
    def test_the_current_block_excludes_the_old_pipeline(self) -> None:
        report = divergence_report(_old_wide(100) + _new_tight(20, index=100))
        self.assertEqual(report["n"], 120)
        self.assertEqual(report["current"]["n"], 20)

    def test_a_fixed_problem_does_not_show_up_in_the_current_block(self) -> None:
        """The whole point: 40pt gaps in the old data, 2pt gaps now."""
        report = divergence_report(_old_wide(100) + _new_tight(20, index=100))
        self.assertGreater(report["medianGapPct"], 15)
        self.assertLess(report["current"]["medianGapPct"], 5)
        self.assertEqual(report["current"]["shareOver15Pct"], 0.0)

    def test_the_pooled_block_is_still_reported(self) -> None:
        """Kept as the trend. Removing it would hide that this ever improved."""
        report = divergence_report(_old_wide(100) + _new_tight(20, index=100))
        for key in ("medianGapPct", "meanGapPct", "shareOver15Pct", "fadesMarket"):
            self.assertIn(key, report)

    def test_no_current_picks_yields_an_empty_block_not_a_missing_one(self) -> None:
        """A caller reading report["current"]["n"] must not raise."""
        report = divergence_report(_old_wide(30))
        self.assertEqual(report["current"], {"n": 0})

    def test_no_paired_observations_at_all(self) -> None:
        unpaired = [
            Observation(event_id="x", league="mlb", date="2026-07-01", home_won=1,
                        model=0.6, market=None, published=0.6, current_pipeline=True)
        ]
        self.assertEqual(divergence_report(unpaired)["n"], 0)


class FadeReportingTests(unittest.TestCase):
    """A fade rate without an interval invites a verdict the sample cannot support."""

    def _fades(self, wins: int, losses: int):
        """Picks where model and market disagree, with a known record."""
        rows = [_obs(0.60, 0.40, 1, current=True, index=i) for i in range(wins)]
        rows += [_obs(0.60, 0.40, 0, current=True, index=wins + i) for i in range(losses)]
        return rows

    def test_the_fade_rate_carries_a_standard_error(self) -> None:
        fade = divergence_report(self._fades(9, 9))["current"]["fadesMarket"]
        self.assertEqual(fade["picks"], 18)
        self.assertAlmostEqual(fade["winPct"], 50.0, places=1)
        # sqrt(0.25/18) * 100 = 11.8, wide enough to span break-even either way.
        self.assertAlmostEqual(fade["stdErrPct"], 11.8, delta=0.2)

    def test_the_error_shrinks_as_the_sample_grows(self) -> None:
        small = divergence_report(self._fades(9, 9))["current"]["fadesMarket"]
        large = divergence_report(self._fades(900, 900))["current"]["fadesMarket"]
        self.assertGreater(small["stdErrPct"], large["stdErrPct"])

    def test_the_fade_share_is_reported_not_just_the_count(self) -> None:
        """116 picks means nothing without knowing it was out of 590."""
        report = divergence_report(self._fades(5, 5) + _new_tight(90, index=500))
        current = report["current"]
        self.assertEqual(current["fadesMarket"]["picks"], 10)
        self.assertAlmostEqual(current["fadesMarket"]["sharePct"], 10.0, places=1)

    def test_no_fades_does_not_divide_by_zero(self) -> None:
        fade = divergence_report(_new_tight(20))["current"]["fadesMarket"]
        self.assertEqual(fade["picks"], 0)
        self.assertIsNone(fade["winPct"])
        self.assertIsNone(fade["stdErrPct"])

    def test_agreeing_and_fading_partition_the_picks(self) -> None:
        report = divergence_report(self._fades(5, 5) + _new_tight(40, index=500))["current"]
        self.assertEqual(
            report["fadesMarket"]["picks"] + report["agreesWithMarket"]["picks"],
            report["n"],
        )


if __name__ == "__main__":
    unittest.main()
