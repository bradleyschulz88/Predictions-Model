"""What shopping across books is worth, recorded so it can be measured.

The build already shops -- `_best_price_for_side` takes the best quote across
every book on the game. What it never did was record the alternatives, so the
value of shopping existed for a few microseconds inside one build and was then
discarded. Asked on 13 Aug 2026 how much line shopping was worth on this board,
the honest answer was that nothing in the repository could say.

It matters because coverage is uneven. The Odds API emits every book it holds,
so those games are genuinely shopped. ESPN core reports one book -- every
pricing line in the build logs reads "via DraftKings" -- and SportsBookReview
is a single board. On those games there is nothing to shop and no record of
what is being left behind.

The unit is deliberate. `gainPct` is implied-probability points, the same scale
as closing line value, which currently runs at a median of -0.4 points. A
shopping gain of one point would more than cover what the model loses to the
close, mechanically and with no modelling risk.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mlb_predictions import quote_spread  # noqa: E402


def _lines(*odds, side: str = "home"):
    return [{"viewType": "MoneyLine", "currentLine": {side: str(value)}} for value in odds]


class QuoteSpreadTests(unittest.TestCase):
    def test_it_reports_the_gap_between_best_and_median(self) -> None:
        spread = quote_spread(_lines(-120, -125, -130), "home")
        self.assertEqual(spread["books"], 3)
        self.assertEqual(spread["best"], -120)
        self.assertEqual(spread["median"], -125)
        self.assertAlmostEqual(spread["gainPct"], 1.01, places=2)

    def test_best_means_the_biggest_payout_not_the_biggest_number(self) -> None:
        """-110 beats -150, and +120 beats both. The classic sign trap."""
        self.assertEqual(quote_spread(_lines(-150, -110, 120), "home")["best"], 120)

    def test_a_single_book_reports_zero_rather_than_nothing(self) -> None:
        """"One book, nothing to gain" is a finding; a missing record is not.

        This is the ESPN case, which is most of the board, and the whole reason
        for measuring: those games cannot be shopped at all.
        """
        spread = quote_spread(_lines(-130), "home")
        self.assertEqual(spread["books"], 1)
        self.assertEqual(spread["gainPct"], 0.0)

    def test_two_books_use_the_worse_one_as_the_stand_in(self) -> None:
        """There is no middle quote in a pair, and averaging them would
        understate what not shopping costs -- the alternative to taking the
        best of two is taking the other one."""
        spread = quote_spread(_lines(-120, -140), "home")
        self.assertEqual(spread["median"], -140)
        self.assertGreater(spread["gainPct"], 3.0)

    def test_underdog_prices_are_handled_the_right_way_round(self) -> None:
        spread = quote_spread(_lines(150, 140, 130), "home")
        self.assertEqual(spread["best"], 150)
        self.assertEqual(spread["median"], 140)
        self.assertGreater(spread["gainPct"], 0)

    def test_an_unpriced_game_reports_nothing(self) -> None:
        self.assertIsNone(quote_spread([], "home"))
        self.assertIsNone(quote_spread(_lines(-120), "draw"))

    def test_it_shares_the_outlier_guard_with_the_price_selector(self) -> None:
        """A +575 beside a -153 was once selected as "best" and published as
        +278.9% EV. The spread must not resurrect it by another route."""
        from mlb_predictions import _best_price_for_side

        lines = _lines(-153, -150, 575)
        self.assertEqual(quote_spread(lines, "home")["best"], _best_price_for_side(lines, "home"))
        self.assertNotEqual(quote_spread(lines, "home")["best"], 575)

    def test_the_best_price_always_agrees_with_the_selector(self) -> None:
        """Two functions picking "the" price from one list is what caused the
        61% grading drift. There is one selector, and this holds it."""
        from mlb_predictions import _best_price_for_side

        for odds in ((-120, -125, -130), (150, 140), (-200,), (-110, 105, -115)):
            with self.subTest(odds=odds):
                lines = _lines(*odds)
                self.assertEqual(
                    quote_spread(lines, "home")["best"], _best_price_for_side(lines, "home")
                )


class SpreadIsRecordedTests(unittest.TestCase):
    """Pinned at pick time, like openingOdds. After the game the books are gone."""

    def test_the_tracker_records_it(self) -> None:
        source = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        self.assertIn('"priceSpread"', source)

    def test_it_is_written_once_and_never_overwritten(self) -> None:
        source = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        self.assertIn('existing.get("priceSpread") or quote_spread(', source)


if __name__ == "__main__":
    unittest.main()
