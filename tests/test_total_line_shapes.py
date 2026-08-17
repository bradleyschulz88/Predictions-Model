"""Two conventions for a total's price, and each parser knew only one.

Found 17 Aug 2026 while checking whether an odd-looking branch in
`compute_total_implied_probabilities` was reachable. It was not, and neither
was the function -- but the reason it was unreachable is a parsing bug that
would have bitten the moment anyone wired the function up.

A Total line's price arrives in one of two shapes:

    "o8.5 (-108)"   ESPN core, the Odds API and ESPN enrichment all fold the
                    line and its price into one display string.
    -108            SportsBookReview keeps them in separate numeric fields.

`_line_odds_value` reads only the second: `int("o8.5 (-108)")` raises, the key
is skipped, and it returns None. `extract_total_price` reads only the first,
via the parenthetical regex. So the two functions accepted disjoint inputs, and
every current producer writes the shape the probability parser could not read.
Measured before the fix:

    shape                 _line_odds_value   compute()   extract_price()
    "o8.5 (-108)"                    None        None              -108
    -108                             -108     over 49.6            None

`compute_total_implied_probabilities` has no callers -- it arrived with the
initial import and never gained one -- so nothing was broken in production.
That is exactly what makes it worth fixing rather than leaving: wired up as-is
it would return None on every game, and "no market total anywhere" reads as a
data gap, not as a parser that cannot see.

Kept rather than deleted because `predict_total` starts from a flat 0.5 and
never anchors to the market. Supplying that anchor is this function's job, and
it is the same defect the moneyline path was rebuilt to remove.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlb_predictions import (
    _total_side_price,
    compute_total_implied_probabilities,
    extract_total_price,
)

# Every shape a producer in this repo actually writes.
SHAPES = {
    "espn core / odds api": {"over": "o8.5 (-108)", "under": "u8.5 (-112)"},
    "espn enrichment": {"over": "8.5 (-108)", "under": "8.5 (-112)"},
    "sbr numeric": {"over": -108, "under": -112},
    "sbr string fields": {"overOdds": "-108", "underOdds": "-112"},
}


def _lines(current: dict[str, object]) -> list[dict[str, object]]:
    return [{"viewType": "Total", "sportsbook": "Book", "currentLine": current}]


class SidePriceTests(unittest.TestCase):
    def test_every_producer_shape_yields_the_same_price(self) -> None:
        for name, current in SHAPES.items():
            with self.subTest(shape=name):
                self.assertEqual(_total_side_price(current, "over"), -108, name)
                self.assertEqual(_total_side_price(current, "under"), -112, name)

    def test_a_line_with_no_price_yields_none(self) -> None:
        self.assertIsNone(_total_side_price({"over": "o8.5", "under": "u8.5"}, "over"))

    def test_an_absent_side_yields_none(self) -> None:
        self.assertIsNone(_total_side_price({}, "over"))

    def test_a_price_from_the_invalid_band_is_not_read(self) -> None:
        self.assertIsNone(_total_side_price({"over": "o8.5 (0)"}, "over"))
        self.assertIsNone(_total_side_price({"over": 50}, "over"))


class ImpliedProbabilityTests(unittest.TestCase):
    def test_every_producer_shape_now_produces_a_market_read(self) -> None:
        for name, current in SHAPES.items():
            with self.subTest(shape=name):
                result = compute_total_implied_probabilities(_lines(current))
                self.assertIsNotNone(result, f"{name} produced nothing")
                self.assertEqual(result["booksUsed"], 1)
                self.assertAlmostEqual(result["overPct"], 49.6, places=1)
                self.assertAlmostEqual(result["underPct"], 50.4, places=1)

    def test_the_two_sides_sum_to_one_hundred(self) -> None:
        """The point of the exercise is a de-vigged pair, not two raw prices."""
        result = compute_total_implied_probabilities(_lines(SHAPES["espn core / odds api"]))
        self.assertAlmostEqual(result["overPct"] + result["underPct"], 100.0, places=1)

    def test_books_are_averaged_rather_than_first_one_wins(self) -> None:
        lines = _lines({"over": "o8.5 (-130)", "under": "u8.5 (+110)"})
        lines += _lines({"over": "o8.5 (+110)", "under": "u8.5 (-130)"})
        result = compute_total_implied_probabilities(lines)
        self.assertEqual(result["booksUsed"], 2)
        self.assertAlmostEqual(result["overPct"], 50.0, places=1)

    def test_an_unpriced_line_contributes_nothing(self) -> None:
        self.assertIsNone(compute_total_implied_probabilities(_lines({"over": "o8.5", "under": "u8.5"})))

    def test_a_one_sided_line_is_skipped_rather_than_halved(self) -> None:
        self.assertIsNone(compute_total_implied_probabilities(_lines({"over": "o8.5 (-108)"})))

    def test_moneyline_and_spread_rows_are_ignored(self) -> None:
        lines = [
            {"viewType": "MoneyLine", "currentLine": {"home": -150, "away": 130}},
            {"viewType": "Spread", "currentLine": {"home": "-1.5 (-110)", "away": "+1.5 (-110)"}},
        ]
        self.assertIsNone(compute_total_implied_probabilities(lines))

    def test_nothing_at_all_is_handled(self) -> None:
        self.assertIsNone(compute_total_implied_probabilities([]))
        self.assertIsNone(compute_total_implied_probabilities(None))


class AgreementWithTheOtherParserTests(unittest.TestCase):
    """The two functions read the same field and must not disagree about it."""

    def test_both_read_the_same_price_from_the_same_line(self) -> None:
        for name, current in SHAPES.items():
            price = extract_total_price(_lines(current), "over")
            if price is None:
                # extract_total_price only ever understood the parenthetical
                # form; that asymmetry is deliberate and is what pick pricing
                # relies on. Where it does read a price, it must be the same one.
                continue
            with self.subTest(shape=name):
                self.assertEqual(price, _total_side_price(current, "over"))


if __name__ == "__main__":
    unittest.main()
