"""American odds between -100 and +100 do not exist, and the build believed them.

Found 17 Aug 2026 by fuzzing the odds helpers rather than by reading them. The
notation is two half-scales that meet at even money: on the plus side the
number is what a 100 stake wins, on the minus side it is what must be staked to
win 100, so +100 and -100 are the same price and nothing lies between. No book
quotes -50.

The arithmetic never checked:

       odds    implied    decimal
        -50     0.3333     3.0000
         -5     0.0476    21.0000
         -1     0.0099   101.0000
          0     1.0000     1.0000   <- an impossible probability
          5     0.9524     1.0500

Two of those are worse than merely wrong. `0` implies certainty. And `-1`
returns decimal 101.0 -- the largest payout either branch can produce -- while
`_best_price_for_side` selects with `max(quotes, key=american_to_decimal)`. A
misparsed field does not degrade a price, it *wins the shop*, and publishes an
enormous fake edge off it.

That is the same failure the outlier guard was written for after a +575 quote
beside a -153 published +278.9% EV where the truth was -7.2%. The guard cannot
catch this one: `_usable_quotes` opened with `if len(quotes) < 2: return quotes`,
and a single-book game -- every ESPN-core game, every SBR game -- carries
exactly one quote. Measured before the fix:

    lone -1        -> [-1]        (returned unexamined)
    -110 and -1    -> []          (pair, both discarded)
    -110,-105,-1   -> [-110,-105] (median available, outlier dropped)
    best from -1   -> -1

So the guard worked precisely where the data was rich enough not to need it.

The fix rejects the band at every boundary a price enters rather than at the
one place the exploit was found, because the paths a bad value can reach are
not worth re-deriving each time a parser is added.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.odds_api import _decimal_to_american
from espn_odds import _american
from market import assess_price, is_valid_american_odds
from mlb_predictions import (
    _best_price_for_side,
    _line_odds_value,
    _moneyline_from_line,
    _usable_quotes,
    american_odds_to_implied,
    extract_spread_price,
    extract_total_price,
    quote_spread,
)

# Every value in the open interval, plus the endpoints that ARE valid.
INVALID = (0, -1, 1, -5, 5, -50, 50, -99, 99, -99.9, 99.5)
VALID = (-100, 100, -110, -155, 120, 575, -1200)


class ValidityPredicateTests(unittest.TestCase):
    def test_the_band_is_rejected(self) -> None:
        for odds in INVALID:
            self.assertFalse(is_valid_american_odds(odds), f"{odds} is not a price")

    def test_the_endpoints_are_kept(self) -> None:
        """+100 and -100 are both even money -- decimal 2.0 -- and both real."""
        for odds in VALID:
            self.assertTrue(is_valid_american_odds(odds), f"{odds} is a real price")

    def test_strings_are_accepted_in_the_shapes_the_feeds_send(self) -> None:
        self.assertTrue(is_valid_american_odds("+120"))
        self.assertTrue(is_valid_american_odds("-110"))
        self.assertFalse(is_valid_american_odds("-50"))

    def test_nothing_that_is_not_a_number_gets_through(self) -> None:
        for value in (None, "", "abc", "EVEN", float("nan"), float("inf"), float("-inf"), [], {}):
            self.assertFalse(is_valid_american_odds(value), f"{value!r} is not a price")


class ImpliedProbabilityTests(unittest.TestCase):
    """The function that returned 1.0 for zero."""

    def test_the_band_yields_no_probability_at_all(self) -> None:
        for odds in INVALID:
            self.assertIsNone(american_odds_to_implied(odds), f"{odds} implied nothing")

    def test_zero_specifically_no_longer_implies_certainty(self) -> None:
        self.assertIsNone(american_odds_to_implied(0))

    def test_real_prices_are_unchanged(self) -> None:
        self.assertAlmostEqual(american_odds_to_implied(-150), 0.6, places=2)
        self.assertAlmostEqual(american_odds_to_implied(130), 0.4348, places=3)

    def test_both_forms_of_even_money_agree(self) -> None:
        self.assertEqual(american_odds_to_implied(100), american_odds_to_implied(-100))
        self.assertAlmostEqual(american_odds_to_implied(100), 0.5)


class LoneQuoteTests(unittest.TestCase):
    """The path the outlier guard could not see.

    `_usable_quotes` returned early on fewer than two quotes, and a single-book
    game is the common case, not the exotic one.
    """

    def test_a_single_bad_quote_no_longer_walks_through(self) -> None:
        for odds in INVALID:
            self.assertEqual(_usable_quotes([odds]), [], f"lone {odds} survived")

    def test_a_single_good_quote_still_does(self) -> None:
        self.assertEqual(_usable_quotes([-110]), [-110])

    def test_a_bad_quote_beside_a_good_one_costs_only_itself(self) -> None:
        """Dropping the invalid value first leaves a lone valid quote, which is
        a priced game. Discarding both would be the two-book disagreement rule,
        and this is not a disagreement -- one of them is simply not a price."""
        self.assertEqual(_usable_quotes([-110, -1]), [-110])

    def test_the_outlier_rule_still_applies_among_valid_quotes(self) -> None:
        self.assertEqual(_usable_quotes([-110, -105, 575]), [-110, -105])
        self.assertEqual(_usable_quotes([-153, 575]), [])


def _moneyline(book: str, home: object, away: object = -110) -> dict[str, object]:
    return {
        "viewType": "MoneyLine",
        "sportsbook": book,
        "currentLine": {"home": home, "away": away},
    }


class BestPriceTests(unittest.TestCase):
    """The consequence that made this worth fixing rather than noting."""

    def test_a_lone_misparsed_price_is_not_selected_as_the_best_one(self) -> None:
        for odds in INVALID:
            self.assertIsNone(
                _best_price_for_side([_moneyline("A", odds)], "home"),
                f"{odds} was still shoppable",
            )

    def test_the_most_dangerous_value_is_covered(self) -> None:
        """-1 is decimal 101.0, the largest payout the conversion can return,
        so it beats every genuine quote it is compared against."""
        lines = [_moneyline("A", -150), _moneyline("B", -140), _moneyline("C", -1)]
        self.assertEqual(_best_price_for_side(lines, "home"), -140)

    def test_shopping_between_real_books_is_untouched(self) -> None:
        lines = [_moneyline("A", -150), _moneyline("B", -140)]
        self.assertEqual(_best_price_for_side(lines, "home"), -140)

    def test_the_spread_record_reports_nothing_rather_than_a_fake_gain(self) -> None:
        self.assertIsNone(quote_spread([_moneyline("A", 0)], "home"))
        spread = quote_spread([_moneyline("A", -150), _moneyline("B", -140)], "home")
        self.assertEqual(spread["books"], 2)


class ExpectedValueTests(unittest.TestCase):
    def test_no_price_in_the_band_is_ever_assessed(self) -> None:
        for odds in INVALID:
            self.assertIsNone(assess_price(0.55, odds), f"{odds} produced an EV")

    def test_a_real_price_still_is(self) -> None:
        priced = assess_price(0.55, -110)
        self.assertIsNotNone(priced)
        self.assertEqual(priced["odds"], -110)


class ParsingBoundaryTests(unittest.TestCase):
    """Each place a price enters the build, guarded at the door."""

    def test_the_line_reader_skips_the_band(self) -> None:
        for odds in INVALID:
            self.assertIsNone(_line_odds_value({"home": odds}, "home", "homeOdds"))

    def test_a_bad_first_key_does_not_shadow_a_good_second_one(self) -> None:
        """ESPN `home` and SBR `homeOdds` can both be present; junk in the one
        tried first must not cost the other."""
        self.assertEqual(_line_odds_value({"home": 0, "homeOdds": -115}, "home", "homeOdds"), -115)

    def test_a_devigged_line_is_not_built_from_one(self) -> None:
        self.assertIsNone(_moneyline_from_line(_moneyline("A", 0)))
        self.assertIsNotNone(_moneyline_from_line(_moneyline("A", -150)))

    def test_the_parenthetical_total_price_is_checked(self) -> None:
        lines = [{"viewType": "Total", "currentLine": {"over": "o8.5 (-5)", "under": "u8.5 (-110)"}}]
        self.assertIsNone(extract_total_price(lines, "over"))
        self.assertEqual(extract_total_price(lines, "under"), -110)

    def test_the_parenthetical_spread_price_is_checked(self) -> None:
        lines = [{"viewType": "Spread", "currentLine": {"home": "-1.5 (0)", "away": "+1.5 (-110)"}}]
        self.assertIsNone(extract_spread_price(lines, "home"))
        self.assertEqual(extract_spread_price(lines, "away"), -110)

    def test_the_espn_parser_rejects_more_than_just_zero(self) -> None:
        """It used `int(value) or None`, which caught 0 and nothing else."""
        for odds in INVALID:
            self.assertIsNone(_american(odds), f"ESPN parser passed {odds}")
        self.assertEqual(_american(-110), -110)
        self.assertEqual(_american("+120"), 120)

    def test_espn_even_money_still_parses(self) -> None:
        for text in ("EVEN", "EV", "PK"):
            self.assertEqual(_american(text), 100)

    def test_an_infinite_espn_price_costs_one_field_not_the_run(self) -> None:
        self.assertIsNone(_american("1e400"))
        self.assertIsNone(_american(float("inf")))

    def test_the_odds_api_conversion_cannot_emit_the_band(self) -> None:
        """Decimal is a continuous scale with no gap, so the conversion lands
        in the band only through a rounding or non-finite path."""
        for decimal in (1.0, 1.000001, 1.5, 1.91, 2.0, 2.01, 6.75, float("inf"), float("nan")):
            converted = _decimal_to_american(decimal)
            if converted is not None:
                self.assertTrue(
                    is_valid_american_odds(converted),
                    f"decimal {decimal} converted to {converted}",
                )
        self.assertEqual(_decimal_to_american(2.0), 100)
        self.assertEqual(_decimal_to_american(1.91), -110)


class DashboardTests(unittest.TestCase):
    """The same formula, unguarded, in the page."""

    def test_the_page_declines_to_convert_the_band(self) -> None:
        board = (ROOT / "dashboard" / "board.js").read_text(encoding="utf-8")
        self.assertIn("if (odds > -100 && odds < 100) return null;", board)


if __name__ == "__main__":
    unittest.main()
