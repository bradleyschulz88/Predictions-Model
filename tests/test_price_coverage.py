"""A degrading price source must be heard from the build, not the board.

The existing odds check fires only when a league prices exactly zero games, so
a source that went from 15/15 to 3/15 was invisible until the board looked thin
days later. This adds the partial case.

The trap is the same one the predictor-coverage check fell into: books post
lines about a day out, so a slate built a week ahead legitimately has no prices
at all. Measured on the 2026-08-05 build, MLB priced 15/15 for 08-01 through
08-04 and 0/11 for 08-06 -- complete for every date a book had time to post,
empty for the ones it had not. Counting the future dates would fire this
warning on every single build.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_coverage import (  # noqa: E402
    MIN_SLATE_FOR_PRICING_WARNING,
    PRICE_EXPECTED_WITHIN_HOURS,
    _price_should_exist,
    coverage_warnings,
    summarize_coverage,
)

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def _game(*, priced: bool, hours_ahead: float = 2.0, final: bool = False,
          live: bool = False, voided: bool = False) -> dict:
    lines = []
    if priced:
        lines = [{
            "sportsbook": "TestBook",
            "viewType": "MoneyLine",
            "currentLine": {"homeOdds": -120, "awayOdds": 100},
        }]
    return {
        "league": "mlb",
        "isFinal": final,
        "isLive": live,
        "isVoided": voided,
        "startDate": (NOW + timedelta(hours=hours_ahead)).isoformat(),
        "lines": lines,
        "enrichment": {},
    }


def _price_warnings(summary: dict, league: str = "mlb") -> list[str]:
    """Only the partial-degradation warning.

    It shares "should have posted by now" with the total-failure warning, so
    the filter keys on "priced only", which is unique to this one.
    """
    return [w for w in coverage_warnings({league: summary}) if "priced only" in w]


class PriceEligibilityTests(unittest.TestCase):
    def test_a_game_under_way_should_have_had_a_price(self) -> None:
        self.assertTrue(_price_should_exist(_game(priced=False, live=True), now=NOW))

    def test_a_finished_game_should_have_had_a_price(self) -> None:
        self.assertTrue(_price_should_exist(_game(priced=False, final=True), now=NOW))

    def test_a_game_inside_the_posting_horizon_counts(self) -> None:
        near = _game(priced=False, hours_ahead=PRICE_EXPECTED_WITHIN_HOURS - 1)
        self.assertTrue(_price_should_exist(near, now=NOW))

    def test_a_game_beyond_the_horizon_does_not_count(self) -> None:
        """MLB 08-06 and 08-07 were 0/11 and 0/15 and nothing was wrong."""
        far = _game(priced=False, hours_ahead=PRICE_EXPECTED_WITHIN_HOURS + 12)
        self.assertFalse(_price_should_exist(far, now=NOW))

    def test_a_voided_game_is_never_expected_to_carry_a_price(self) -> None:
        self.assertFalse(_price_should_exist(_game(priced=False, voided=True), now=NOW))

    def test_a_game_with_no_start_time_still_counts(self) -> None:
        """A feed that stops emitting start times must not silence the check."""
        game = _game(priced=False)
        game.pop("startDate")
        self.assertTrue(_price_should_exist(game, now=NOW))

    def test_an_unparseable_start_time_still_counts(self) -> None:
        game = _game(priced=False)
        game["startDate"] = "not a date"
        self.assertTrue(_price_should_exist(game, now=NOW))


class PriceCoverageWarningTests(unittest.TestCase):
    def test_a_healthy_slate_is_silent(self) -> None:
        summary = summarize_coverage([_game(priced=True) for _ in range(15)], now=NOW)
        self.assertEqual(summary["pricedSharePct"], 100.0)
        self.assertEqual(_price_warnings(summary), [])

    def test_a_degraded_source_fires(self) -> None:
        games = [_game(priced=True) for _ in range(3)]
        games += [_game(priced=False) for _ in range(12)]
        summary = summarize_coverage(games, now=NOW)
        self.assertEqual(summary["pricedSharePct"], 20.0)
        warnings = _price_warnings(summary)
        self.assertEqual(len(warnings), 1)
        self.assertIn("3/15", warnings[0])

    def test_one_missing_book_does_not_fire(self) -> None:
        """14/15 is the healthy case observed on 2026-08-05, not a fault."""
        games = [_game(priced=True) for _ in range(14)] + [_game(priced=False)]
        summary = summarize_coverage(games, now=NOW)
        self.assertEqual(_price_warnings(summary), [])

    def test_a_future_slate_never_fires(self) -> None:
        """The whole board carries a week of slates; none may raise this."""
        far = PRICE_EXPECTED_WITHIN_HOURS + 24
        summary = summarize_coverage([_game(priced=False, hours_ahead=far) for _ in range(15)], now=NOW)
        self.assertEqual(summary["pricedEligible"], 0)
        self.assertIsNone(summary["pricedSharePct"])
        self.assertEqual(_price_warnings(summary), [])

    def test_a_tiny_slate_does_not_fire(self) -> None:
        """One of three is 33% and means nothing."""
        games = [_game(priced=True)] + [_game(priced=False) for _ in range(2)]
        summary = summarize_coverage(games, now=NOW)
        self.assertLess(summary["pricedEligible"], MIN_SLATE_FOR_PRICING_WARNING)
        self.assertEqual(_price_warnings(summary), [])

    def test_zero_priced_is_left_to_the_existing_check(self) -> None:
        """Not double-reported: the total-failure case already has a warning."""
        summary = summarize_coverage([_game(priced=False) for _ in range(15)], now=NOW)
        self.assertEqual(_price_warnings(summary), [])
        total = [w for w in coverage_warnings({"mlb": summary}) if "odds source configured" in w]
        self.assertEqual(len(total), 1)

    def test_a_mixed_slate_only_counts_the_games_a_book_had_time_for(self) -> None:
        games = [_game(priced=True) for _ in range(6)]
        games += [_game(priced=False, hours_ahead=PRICE_EXPECTED_WITHIN_HOURS + 24)
                  for _ in range(20)]
        summary = summarize_coverage(games, now=NOW)
        self.assertEqual(summary["pricedEligible"], 6)
        self.assertEqual(summary["pricedSharePct"], 100.0)
        self.assertEqual(_price_warnings(summary), [])

    def test_a_league_with_no_odds_source_is_not_warned(self) -> None:
        """AFL carries no configured odds source, so silence is correct.

        Checked against _expects_odds rather than assumed: epl DOES expect
        odds, which an earlier version of this test got wrong.
        """
        from data_coverage import _expects_odds

        self.assertFalse(_expects_odds("afl"))
        summary = summarize_coverage([_game(priced=False) for _ in range(15)], now=NOW)
        self.assertEqual(
            [w for w in coverage_warnings({"afl": summary}) if "priced" in w],
            [],
        )


if __name__ == "__main__":
    unittest.main()
