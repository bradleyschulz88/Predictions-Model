"""Tests for odds arithmetic, de-vigging and expected value."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market import (  # noqa: E402
    american_to_decimal,
    american_to_implied,
    assess_price,
    break_even_probability,
    decimal_to_american,
    devig_power,
    devig_proportional,
    expected_value,
    kelly_fraction,
)


class ConversionTests(unittest.TestCase):
    def test_negative_odds_to_decimal(self) -> None:
        self.assertAlmostEqual(american_to_decimal(-155), 1.6452, places=3)

    def test_positive_odds_to_decimal(self) -> None:
        self.assertAlmostEqual(american_to_decimal(135), 2.35, places=3)

    def test_even_money_is_two(self) -> None:
        self.assertAlmostEqual(american_to_decimal(100), 2.0)

    def test_implied_probability(self) -> None:
        self.assertAlmostEqual(american_to_implied(-300), 0.75)
        self.assertAlmostEqual(american_to_implied(300), 0.25)

    def test_round_trips(self) -> None:
        for odds in (-300, -155, -110, 110, 135, 250):
            self.assertEqual(decimal_to_american(american_to_decimal(odds)), odds)


class DevigTests(unittest.TestCase):
    def _raw(self, home, away):
        return [american_to_implied(home), american_to_implied(away)]

    def test_both_methods_sum_to_one(self) -> None:
        raw = self._raw(-300, 250)
        for fair in (devig_proportional(raw), devig_power(raw)):
            self.assertAlmostEqual(sum(fair), 1.0, places=6)

    def test_identical_on_a_pick_em(self) -> None:
        """With symmetric prices there is no skew to correct."""
        raw = self._raw(-110, -110)
        self.assertAlmostEqual(devig_power(raw)[0], devig_proportional(raw)[0], places=6)

    def test_power_lifts_the_favourite_on_a_lopsided_market(self) -> None:
        """Books load margin onto the longshot; proportional understates the favourite."""
        raw = self._raw(-300, 250)
        self.assertGreater(devig_power(raw)[0], devig_proportional(raw)[0])

    def test_handles_a_three_way_market(self) -> None:
        raw = [american_to_implied(x) for x in (150, 240, 190)]
        fair = devig_power(raw)
        self.assertEqual(len(fair), 3)
        self.assertAlmostEqual(sum(fair), 1.0, places=6)

    def test_no_overround_is_left_alone(self) -> None:
        fair = devig_power([0.4, 0.6])
        self.assertAlmostEqual(fair[0], 0.4, places=6)

    def test_degenerate_input_does_not_explode(self) -> None:
        self.assertEqual(len(devig_power([0.0, 0.0])), 2)


class ExpectedValueTests(unittest.TestCase):
    def test_fair_price_has_zero_edge(self) -> None:
        self.assertAlmostEqual(expected_value(0.5, 100), 0.0, places=9)

    def test_break_even_matches_the_price(self) -> None:
        self.assertAlmostEqual(break_even_probability(-155), 155 / 255, places=6)
        self.assertAlmostEqual(expected_value(break_even_probability(-155), -155), 0.0, places=9)

    def test_same_edge_pays_more_on_a_longshot(self) -> None:
        """The point of EV: percentage points of edge are not comparable across prices."""
        fav = expected_value(break_even_probability(-300) + 0.03, -300)
        dog = expected_value(break_even_probability(250) + 0.03, 250)
        self.assertGreater(dog, fav * 2)

    def test_negative_edge_is_negative_ev(self) -> None:
        self.assertLess(expected_value(0.50, -155), 0)


class KellyTests(unittest.TestCase):
    def test_no_edge_means_no_stake(self) -> None:
        self.assertAlmostEqual(kelly_fraction(break_even_probability(-155), -155), 0.0, places=9)

    def test_negative_edge_never_stakes(self) -> None:
        self.assertEqual(kelly_fraction(0.4, -155), 0.0)

    def test_bigger_edge_stakes_more(self) -> None:
        self.assertGreater(kelly_fraction(0.70, -155), kelly_fraction(0.62, -155))

    def test_certainty_stakes_everything(self) -> None:
        self.assertAlmostEqual(kelly_fraction(1.0, 100), 1.0, places=6)


class AssessPriceTests(unittest.TestCase):
    def test_reports_a_positive_bet(self) -> None:
        result = assess_price(0.65, -155)
        self.assertTrue(result["positive"])
        self.assertGreater(result["evPct"], 0)
        self.assertGreater(result["kellyPct"], 0)

    def test_reports_a_negative_bet(self) -> None:
        result = assess_price(0.55, -155)
        self.assertFalse(result["positive"])
        self.assertLess(result["evPct"], 0)
        self.assertEqual(result["kellyPct"], 0.0)

    def test_stake_is_scaled_below_full_kelly(self) -> None:
        """Full Kelly assumes the probability is exact, which it never is."""
        result = assess_price(0.65, -155)
        self.assertLess(result["kellyPct"] / 100.0, kelly_fraction(0.65, -155))

    def test_missing_inputs_yield_nothing(self) -> None:
        self.assertIsNone(assess_price(None, -155))
        self.assertIsNone(assess_price(0.6, None))
        self.assertIsNone(assess_price(0.6, 0))

    def test_unparseable_odds_yield_nothing(self) -> None:
        self.assertIsNone(assess_price(0.6, "evens"))

    def test_kelly_probability_overrides_only_the_stake(self) -> None:
        """A calibration band saying this confidence level is really a coin
        flip should shrink the stake without touching the displayed number."""
        result = assess_price(0.85, -110, kelly_probability=0.52)
        self.assertEqual(result["modelPct"], 85.0, "the headline number is unmoved")
        self.assertEqual(result["kellyProbabilityPct"], 52.0)
        self.assertLess(result["kellyPct"], assess_price(0.85, -110)["kellyPct"])

    def test_kelly_probability_can_raise_the_stake_too(self) -> None:
        """Calibration cuts both ways -- a band that has outperformed its own
        number should be allowed to stake more, not just less."""
        baseline = assess_price(0.60, -110)["kellyPct"]
        boosted = assess_price(0.60, -110, kelly_probability=0.75)["kellyPct"]
        self.assertGreater(boosted, baseline)

    def test_no_override_sizes_off_the_model_probability(self) -> None:
        result = assess_price(0.65, -155)
        self.assertEqual(result["kellyProbabilityPct"], result["modelPct"])




class PriceAgreementTests(unittest.TestCase):
    """The price the board advertises must be the price the record grades at.

    _best_price_for_side takes the best moneyline across every book quoting a
    game, and drives EV, Kelly and the board ranking. extract_pick_american_odds
    used to walk the same lines itself and return the FIRST one it found, which
    is a different number whenever books disagree. On the graded record the two
    disagreed 61% of the time, always in the same direction: an edge advertised
    at a price nobody booked. A 60% pick shown at -136 as +4.1% EV was graded at
    -155, where it is -1.3%.
    """

    def _game(self, *home_prices):
        return {"lines": [
            {"viewType": "MoneyLine", "currentLine": {"home": str(p), "away": "+100"}}
            for p in home_prices
        ]}

    def test_the_two_paths_return_the_same_price(self) -> None:
        from accuracy_tracker import extract_pick_american_odds
        from mlb_predictions import _best_price_for_side

        for prices in ((-155, -136), (-110,), (+120, +135, +128)):
            game = self._game(*prices)
            self.assertEqual(
                _best_price_for_side(game["lines"], "home"),
                extract_pick_american_odds(game, "home"),
                msg=f"EV and grading disagree on {prices}",
            )

    def test_grading_takes_the_best_price_not_the_first_listed(self) -> None:
        from accuracy_tracker import extract_pick_american_odds

        # -136 pays better than -155, and is listed second.
        self.assertEqual(extract_pick_american_odds(self._game(-155, -136), "home"), -136)

    def test_an_ev_computed_at_the_graded_price_is_the_published_one(self) -> None:
        """End to end: the number on the card is the number that gets settled."""
        from accuracy_tracker import extract_pick_american_odds
        from mlb_predictions import _best_price_for_side

        game = self._game(-155, -136)
        published = assess_price(0.60, _best_price_for_side(game["lines"], "home"))
        graded_price = extract_pick_american_odds(game, "home")
        self.assertAlmostEqual(
            published["evPct"], expected_value(0.60, graded_price) * 100, places=2,
        )


class OutlierQuoteTests(unittest.TestCase):
    """Books disagree by a point or two, not by tens of points.

    One WNBA game carried a +575 quote beside a -153. "Best price" picked the
    +575 and published an EV of +278.9% where the real figure was -7.2%, because
    a garbage price always looks like the best one.
    """

    def _lines(self, *prices):
        return [{"viewType": "MoneyLine", "currentLine": {"home": str(p)}} for p in prices]

    def test_a_normal_spread_of_prices_is_untouched(self) -> None:
        from mlb_predictions import _best_price_for_side

        self.assertEqual(_best_price_for_side(self._lines(-155, -136), "home"), -136)

    def test_two_wildly_disagreeing_quotes_discard_the_market(self) -> None:
        """With two, there is no way to tell which is wrong. An unpriced game
        is honest; a confidently wrong price is not."""
        from mlb_predictions import _best_price_for_side

        self.assertIsNone(_best_price_for_side(self._lines(-153, 575), "home"))

    def test_with_three_the_median_identifies_the_outlier(self) -> None:
        from mlb_predictions import _best_price_for_side

        self.assertEqual(_best_price_for_side(self._lines(-150, -145, 575), "home"), -145)

    def test_a_lone_quote_is_kept(self) -> None:
        """Nothing to compare it against is not the same as knowing it is bad."""
        from mlb_predictions import _best_price_for_side

        self.assertEqual(_best_price_for_side(self._lines(-155), "home"), -155)

    def test_the_guard_reaches_grading_too(self) -> None:
        from accuracy_tracker import extract_pick_american_odds

        self.assertIsNone(
            extract_pick_american_odds({"lines": self._lines(-153, 575)}, "home")
        )

if __name__ == "__main__":
    unittest.main()
