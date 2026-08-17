"""`predict_total` began at a coin flip and nudged. Now it begins at the price.

The moneyline path was rebuilt to anchor on the market and model the residual.
Totals never were: `over_lean` started at a flat 0.5 -- "no idea" -- and a
series of hand-set constants pushed it around. The market total is the best
available estimate of whether a game goes over, because books move it on the
same information plus the money. Treating a coin flip as equally informative is
how the model ended up two-to-one on one side.

Measured over 324 graded totals before this change:

    picked OVER on 219 of 324   67.6%
    over   104/219   47.5%
    under   58/105   55.2%
    priced hit rate 47.3% +/- 3.3 vs a 52.3% break-even, ROI -8.7%

The under picks doing fine is the tell. When the signals were strong enough to
overcome the drift they were right; the drift itself was losing money.

What this change does and does not do, stated precisely, because the
distinction matters:

  * It fixes the LEVEL. Published confidence for the same signal drops from
    70.0% to 55.0% at a typical nudge, and the number of nudge values clearing
    a 57% publication threshold falls from 44/51 to 22/51.
  * It fixes the SIDE only where the market itself leans -- 14% of a grid over
    realistic prices and reachable nudges. At a market of exactly 50.0% the
    side still follows the sign of the nudge, unchanged.

So the over-bias is reduced, not eliminated. Its remaining source is inside the
nudges, and it cannot be attributed from the graded record because that record
stored only line, side and outcome. This change starts logging marketOverPct,
modelOverPct and the published number so the next few hundred games can say
which term is at fault. That instrumentation is the point as much as the anchor
is.

One genuine one-sided term was found and fixed on the way: two strong records
added 0.08 toward the over for NBA/WNBA/AFL and nothing ever subtracted. A
signal that can only fire one way is a bias with a reason attached.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import grade_total
from mlb_predictions import (
    MAX_TOTAL_RESIDUAL_LOGIT,
    TOTAL_MODEL_WEIGHT,
    _total_market_anchor,
    predict_total,
)


def priced(total: float = 8.5, over: int = -110, under: int = -110) -> list[dict]:
    return [{
        "viewType": "Total", "sportsbook": "Book",
        "currentLine": {"over": f"o{total:g} ({over:+d})", "under": f"u{total:g} ({under:+d})"},
    }]


def unpriced(total: float = 8.5) -> list[dict]:
    """SportsBookReview publishes the number with no odds attached."""
    return [{"viewType": "Total", "sportsbook": "SBR",
             "currentLine": {"over": f"o{total:g}", "under": f"u{total:g}"}}]


GAME = {"league": "mlb", "leagueId": "mlb", "homeTeam": "A", "awayTeam": "B",
        "homeRecord": "50-40", "awayRecord": "48-42"}

ALL_OVER = ({"homeAdvanced": {"runsPerGame": 6.5}, "awayAdvanced": {"runsPerGame": 6.5}},
            dict(GAME, homePitcher={"era": 5.2}, awayPitcher={"era": 5.4}))
ALL_UNDER = ({"homeAdvanced": {"runsPerGame": 3.2}, "awayAdvanced": {"runsPerGame": 3.3}},
             dict(GAME, homePitcher={"era": 2.9}, awayPitcher={"era": 3.1}))


class AnchorTests(unittest.TestCase):
    def test_a_silent_model_publishes_the_market(self) -> None:
        """No signal firing means no disagreement, so the price stands."""
        result = predict_total(GAME, priced(over=-130, under=110), {})
        self.assertEqual(result["overPct"], result["marketOverPct"])

    def test_the_published_number_follows_the_price(self) -> None:
        cheap = predict_total(GAME, priced(over=110, under=-130), {})
        dear = predict_total(GAME, priced(over=-130, under=110), {})
        self.assertLess(cheap["overPct"], 50.0)
        self.assertGreater(dear["overPct"], 50.0)

    def test_the_anchor_is_devigged_not_raw(self) -> None:
        """Two -110s are 52.4% each raw; de-vigged they are 50/50."""
        self.assertAlmostEqual(_total_market_anchor(priced()), 0.5, places=3)

    def test_an_unpriced_line_falls_back_and_says_so(self) -> None:
        enrichment, game = ALL_OVER
        result = predict_total(game, unpriced(), enrichment)
        self.assertFalse(result["marketAnchored"])
        self.assertIsNone(result["marketOverPct"])

    def test_a_priced_line_is_flagged_anchored(self) -> None:
        self.assertTrue(predict_total(GAME, priced(), {})["marketAnchored"])

    def test_no_total_line_at_all_still_returns_nothing(self) -> None:
        self.assertIsNone(predict_total(GAME, [], {}))


class MagnitudeTests(unittest.TestCase):
    """The part of the bias this change actually fixes."""

    def test_a_full_signal_no_longer_reaches_the_old_clamp(self) -> None:
        enrichment, game = ALL_OVER
        result = predict_total(game, priced(), enrichment)
        self.assertEqual(result["modelOverPct"], 70.0, "the old published number")
        self.assertLess(result["overPct"], 60.0, "the new one")

    def test_the_residual_is_capped(self) -> None:
        """No stack of nudges may drag the market further than the cap."""
        enrichment, game = ALL_OVER
        result = predict_total(game, priced(), enrichment)
        from model_fit import logit
        self.assertLessEqual(
            abs(logit(result["overPct"] / 100) - logit(result["marketOverPct"] / 100)),
            MAX_TOTAL_RESIDUAL_LOGIT + 1e-6,
        )

    def test_the_direction_of_the_signal_is_preserved(self) -> None:
        over_enrich, over_game = ALL_OVER
        under_enrich, under_game = ALL_UNDER
        self.assertGreater(predict_total(over_game, priced(), over_enrich)["overPct"], 50.0)
        self.assertLess(predict_total(under_game, priced(), under_enrich)["overPct"], 50.0)

    def test_the_two_directions_are_symmetric_about_the_market(self) -> None:
        over_enrich, over_game = ALL_OVER
        under_enrich, under_game = ALL_UNDER
        up = predict_total(over_game, priced(), over_enrich)["overPct"] - 50.0
        down = 50.0 - predict_total(under_game, priced(), under_enrich)["overPct"]
        self.assertAlmostEqual(up, down, places=1)

    def test_the_weight_is_what_shrinks_it(self) -> None:
        self.assertGreater(TOTAL_MODEL_WEIGHT, 0.0)
        self.assertLess(TOTAL_MODEL_WEIGHT, 1.0)


class OneSidedNudgeTests(unittest.TestCase):
    """Two strong records pushed over; two weak ones pushed nowhere."""

    def _wnba(self, home: str, away: str) -> dict:
        game = {"league": "wnba", "leagueId": "wnba", "homeTeam": "A", "awayTeam": "B",
                "homeRecord": home, "awayRecord": away}
        return predict_total(game, priced(160), {})

    def test_two_strong_records_still_lean_over(self) -> None:
        self.assertGreater(self._wnba("20-4", "19-5")["modelOverPct"], 50.0)

    def test_two_weak_records_now_lean_under(self) -> None:
        self.assertLess(self._wnba("5-19", "4-20")["modelOverPct"], 50.0)

    def test_the_two_are_equal_and_opposite(self) -> None:
        strong = self._wnba("20-4", "19-5")["modelOverPct"] - 50.0
        weak = 50.0 - self._wnba("5-19", "4-20")["modelOverPct"]
        self.assertAlmostEqual(strong, weak, places=6)

    def test_average_records_lean_neither_way(self) -> None:
        self.assertEqual(self._wnba("12-12", "12-12")["modelOverPct"], 50.0)

    def test_the_term_does_not_apply_to_baseball(self) -> None:
        strong = dict(GAME, homeRecord="90-20", awayRecord="88-22")
        self.assertEqual(predict_total(strong, priced(), {})["modelOverPct"], 50.0)


class InstrumentationTests(unittest.TestCase):
    """Without these the weight above can never be checked against outcomes."""

    def test_the_pick_carries_all_three_numbers(self) -> None:
        enrichment, game = ALL_OVER
        result = predict_total(game, priced(over=-120, under=100), enrichment)
        for key in ("marketOverPct", "modelOverPct", "overPct", "marketAnchored"):
            self.assertIn(key, result)
        self.assertNotEqual(result["marketOverPct"], result["modelOverPct"])

    def test_model_over_pct_is_what_the_old_code_would_have_published(self) -> None:
        enrichment, game = ALL_OVER
        self.assertEqual(predict_total(game, priced(), enrichment)["modelOverPct"], 70.0)

    def test_the_graded_row_keeps_them(self) -> None:
        pick = {"line": 8.5, "pickSide": "over", "odds": -110,
                "marketOverPct": 51.0, "modelOverPct": 70.0, "overPct": 55.5,
                "marketAnchored": True}
        graded = grade_total(pick, 6, 5)
        self.assertEqual(graded["outcome"], "win")
        self.assertEqual(graded["marketOverPct"], 51.0)
        self.assertEqual(graded["modelOverPct"], 70.0)
        self.assertEqual(graded["publishedOverPct"], 55.5)
        self.assertTrue(graded["marketAnchored"])

    def test_an_older_row_without_them_still_grades(self) -> None:
        """Every row logged before today lacks these keys."""
        graded = grade_total({"line": 8.5, "pickSide": "under", "odds": -110}, 3, 4)
        self.assertEqual(graded["outcome"], "win")
        self.assertIsNone(graded["marketOverPct"])

    def test_the_log_keeps_the_fields_it_needs_to_grade_them(self) -> None:
        source = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        for key in ("marketOverPct", "modelOverPct", "overPct", "marketAnchored"):
            self.assertIn(f'"{key}"', source, f"{key} is never written to the log")


class UnchangedContractTests(unittest.TestCase):
    """Callers and the page read these; none may disappear."""

    def test_the_published_shape_is_intact(self) -> None:
        result = predict_total(GAME, priced(), {})
        for key in ("line", "pick", "pickSide", "overPct", "underPct", "confidence", "odds", "detail"):
            self.assertIn(key, result)

    def test_the_two_sides_still_sum_to_one_hundred(self) -> None:
        result = predict_total(GAME, priced(over=-135, under=115), {})
        self.assertAlmostEqual(result["overPct"] + result["underPct"], 100.0, places=6)

    def test_confidence_is_the_larger_side(self) -> None:
        for lines in (priced(over=-135, under=115), priced(over=115, under=-135)):
            result = predict_total(GAME, lines, {})
            self.assertAlmostEqual(
                result["confidence"], max(result["overPct"], result["underPct"]), places=6
            )

    def test_the_price_still_comes_from_the_side_taken(self) -> None:
        result = predict_total(GAME, priced(over=-130, under=110), {})
        self.assertEqual(result["pickSide"], "over")
        self.assertEqual(result["odds"], -130)

    def test_the_clamp_still_bounds_the_published_number(self) -> None:
        enrichment, game = ALL_OVER
        result = predict_total(game, priced(over=-400, under=300), enrichment)
        self.assertLessEqual(result["overPct"], 70.0)
        self.assertGreaterEqual(result["overPct"], 30.0)


if __name__ == "__main__":
    unittest.main()
