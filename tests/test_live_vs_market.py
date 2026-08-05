"""The live model must be scorable against the market on the same games.

The evaluation table leads with `model (published)`, which is whatever version
was live when each game was predicted. That pools every model this log has ever
carried, so a model fixed last week still reads as losing to the market for as
long as its own bad history dominates the record -- and it did: published
logloss 0.6766 against the market's 0.6510, while the live model was actually
ahead at 0.6480 against 0.6550 on the same games.

Acting on the pooled number would have meant shrinking a model toward a market
it already beats.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_fit  # noqa: E402
from model_fit import Sample, walk_forward_scores  # noqa: E402


def _samples(n: int, *, market_skill: float, model_skill: float) -> list[Sample]:
    """Games with a genuine true probability, not a signal that gives the answer.

    An earlier version set the label from the sign of the signal, which made the
    outcome a deterministic function of the feature -- so the fitted model won
    no matter how little skill it was given, and the test could not fail.
    Here the label is drawn against a true probability, and each forecaster sees
    that truth attenuated by its own skill.
    """
    out = []
    for i in range(n):
        # Deterministic pseudo-randoms, so the test cannot flake.
        u_truth = (((i * 2654435761) % 997) / 997.0) * 2.0 - 1.0
        u_draw = ((i * 40503) % 991) / 991.0
        u_noise = (((i * 69069) % 983) / 983.0) * 2.0 - 1.0

        true_logit = u_truth * 1.5
        label = 1 if u_draw < model_fit.sigmoid(true_logit) else 0
        out.append(Sample(
            date=f"2026-0{1 + i % 9}-{1 + i % 28:02d}",
            league="mlb",
            label=label,
            values={
                "strengthDiff": true_logit * model_skill + u_noise * (1.0 - model_skill),
                "marketLogit": true_logit * market_skill,
            },
        ))
    return out


class VsMarketBlockTests(unittest.TestCase):
    def test_walk_forward_reports_a_market_head_to_head(self) -> None:
        scores = walk_forward_scores(_samples(400, market_skill=0.8, model_skill=0.8))
        head = scores.get("vsMarket")
        self.assertIsNotNone(head, "walk-forward must score the market on the same games")
        self.assertEqual(head["n"], scores["n"])
        self.assertIn("edge", head)

    def test_edge_is_positive_when_the_model_is_the_better_forecaster(self) -> None:
        scores = walk_forward_scores(_samples(400, market_skill=0.2, model_skill=1.2))
        self.assertGreater(scores["vsMarket"]["edge"], 0)

    def test_edge_is_negative_when_the_market_is_better(self) -> None:
        """The model must be denied the market here, or it cannot lose.

        marketLogit is an anchored feature, so a fitted model that merely
        shrinks an overconfident market already beats the raw line. To test
        that a negative edge is reportable at all, the fit has to run blind.
        """
        scores = walk_forward_scores(
            _samples(400, market_skill=1.0, model_skill=0.02),
            anchored_features=("strengthDiff",),
            standalone_features=("strengthDiff",),
        )
        self.assertLess(scores["vsMarket"]["edge"], 0)

    def test_market_loss_is_computed_per_game_not_carried_over(self) -> None:
        """A walrus inside a conditional reused the previous game's probability.

        It produced 0.714 where the correct answer was 0.655 -- a plausible
        number, wrong by more than the entire edge being measured.
        """
        samples = _samples(400, market_skill=0.9, model_skill=0.9)
        scores = walk_forward_scores(samples)
        head = scores["vsMarket"]

        # Recompute independently from the same definition.
        ordered = sorted(samples, key=lambda s: (s.date, s.league))
        start = len(ordered) // 6
        expected: list[tuple[float, int]] = []
        for fold in range(5):
            split = start * (fold + 1)
            test = ordered[split:split + start]
            for s in test:
                expected.append((s.values["marketLogit"], s.label))
        loss = -sum(
            math.log(max(1e-9, model_fit.sigmoid(m) if y else 1.0 - model_fit.sigmoid(m)))
            for m, y in expected
        ) / len(expected)
        self.assertAlmostEqual(head["marketLogLoss"], round(loss, 4), places=4)

    def test_games_without_a_price_are_excluded(self) -> None:
        samples = _samples(400, market_skill=0.8, model_skill=0.8)
        for s in samples[::2]:
            s.values["marketLogit"] = None
        scores = walk_forward_scores(samples)
        self.assertLess(scores["vsMarket"]["n"], scores["n"])




class LiveHomeBiasTests(unittest.TestCase):
    """Home bias must be measured on the live model, with its uncertainty.

    The published figure read MLB at +6.4pts, a large and specific-looking
    fault. The live model is at +0.4pts on 500 games -- the +6.4 belongs to
    model versions that no longer run. And WNBA's +4.2pts on 95 games is
    inside one standard error, so it is not a finding either.
    """

    def test_a_neutral_model_reports_no_bias(self) -> None:
        scores = walk_forward_scores(_samples(400, market_skill=0.8, model_skill=0.8))
        bias = scores["homeBias"]["mlb"]
        self.assertFalse(bias["significant"])
        self.assertLess(abs(bias["biasPct"]), 1.96 * bias["stdErrPct"])

    def test_a_real_lean_is_flagged_significant(self) -> None:
        """Push every prediction toward home and the check must catch it."""
        samples = _samples(400, market_skill=0.8, model_skill=0.8)
        for s in samples:
            s.values["strengthDiff"] = abs(s.values["strengthDiff"]) + 3.0
            s.values["marketLogit"] = abs(s.values["marketLogit"]) + 3.0
        scores = walk_forward_scores(samples)
        bias = scores["homeBias"]["mlb"]
        self.assertGreater(bias["biasPct"], 0)
        self.assertTrue(bias["significant"])

    def test_standard_error_shrinks_with_sample_size(self) -> None:
        small = walk_forward_scores(_samples(200, market_skill=0.8, model_skill=0.8))
        large = walk_forward_scores(_samples(900, market_skill=0.8, model_skill=0.8))
        self.assertLess(
            large["homeBias"]["mlb"]["stdErrPct"],
            small["homeBias"]["mlb"]["stdErrPct"],
        )

    def test_bias_is_reported_per_league(self) -> None:
        samples = _samples(400, market_skill=0.8, model_skill=0.8)
        for s in samples[::2]:
            s.league = "wnba"
        scores = walk_forward_scores(samples)
        self.assertIn("mlb", scores["homeBias"])
        self.assertIn("wnba", scores["homeBias"])


if __name__ == "__main__":
    unittest.main()
