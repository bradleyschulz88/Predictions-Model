"""Tests for the ESPN core-API odds source.

This exists because SBR prices neither WNBA nor AFL: WNBA comes back with
spread and total but never a moneyline, and AFL has no board at all. Between
them that is about a third of the model's graded history running with no
market to anchor to.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import unittest
from unittest.mock import patch

from espn_odds import (
    NON_BOOK_PROVIDERS,
    build_odds_url,
    fetch_event_odds,
    fill_missing_moneylines,
    parse_core_odds,
)
from mlb_predictions import compute_implied_probabilities, has_moneyline_lines
from sports_config import list_league_ids


def book(name: str, home: int, away: int, *, provider_id: str = "58", draw: int | None = None) -> dict:
    item = {
        "provider": {"id": provider_id, "name": name},
        "homeTeamOdds": {"moneyLine": home},
        "awayTeamOdds": {"moneyLine": away},
    }
    if draw is not None:
        item["drawOdds"] = {"moneyLine": draw}
    return item


class UrlTests(unittest.TestCase):
    def test_every_league_builds_a_url(self) -> None:
        """espn_path is already "{sport}/{league}", which is what the core API
        wants -- so one config drives both scoreboard and odds."""
        for league in list_league_ids():
            url = build_odds_url(league, "401234567")
            self.assertTrue(url.startswith("https://sports.core.api.espn.com/v2/sports/"))
            self.assertIn("/events/401234567/competitions/401234567/odds", url)

    def test_wnba_and_afl_resolve_correctly(self) -> None:
        """The two leagues this was built for."""
        self.assertIn("/basketball/leagues/wnba/", build_odds_url("wnba", "1"))
        self.assertIn("/australian-football/leagues/afl/", build_odds_url("afl", "1"))

    def test_soccer_keeps_its_dotted_slug(self) -> None:
        self.assertIn("/soccer/leagues/eng.1/", build_odds_url("epl", "1"))


class ParseTests(unittest.TestCase):
    def test_flat_moneyline(self) -> None:
        lines = parse_core_odds({"items": [book("ESPN BET", -140, 120)]})
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["viewType"], "MoneyLine")
        self.assertEqual(lines[0]["currentLine"], {"home": -140, "away": 120})

    def test_nested_current_american(self) -> None:
        """A second shape ESPN serves for the same competition."""
        payload = {"items": [{
            "provider": {"id": "31", "name": "DraftKings"},
            "homeTeamOdds": {"current": {"moneyLine": {"american": "-155"}}},
            "awayTeamOdds": {"current": {"moneyLine": {"american": "+130"}}},
        }]}
        self.assertEqual(parse_core_odds(payload)[0]["currentLine"], {"home": -155, "away": 130})

    def test_three_way_keeps_the_draw(self) -> None:
        lines = parse_core_odds({"items": [book("Bet365", 150, 200, draw=230)]})
        self.assertEqual(lines[0]["currentLine"]["draw"], 230)

    def test_opening_line_is_captured_when_present(self) -> None:
        """Needed for closing-line value, which is currently a 0% stub."""
        item = book("ESPN BET", -140, 120)
        item["open"] = {"homeTeamOdds": {"moneyLine": -135}, "awayTeamOdds": {"moneyLine": 115}}
        self.assertEqual(parse_core_odds({"items": [item]})[0]["openingLine"],
                         {"home": -135, "away": 115})

    def _spread_total_only(self) -> dict:
        return {"items": [{
            "provider": {"id": "59", "name": "Caesars"},
            "spread": -4.5, "overUnder": 165.5,
            "homeTeamOdds": {"spreadOdds": -110}, "awayTeamOdds": {"spreadOdds": -110},
        }]}

    def test_spread_and_total_only_row_yields_no_moneyline(self) -> None:
        """The exact WNBA-on-SBR symptom. A priceless row must not look priced.

        `marketLogit` is built from the moneyline alone, so a row carrying only a
        spread and a total must never produce one.
        """
        views = {line["viewType"] for line in parse_core_odds(self._spread_total_only())}
        self.assertNotIn("MoneyLine", views)

    def test_spread_and_total_are_still_captured(self) -> None:
        """They are useless for the win model and exactly what the card needs.

        Emitting only MoneyLine is why totals and spreads never appeared on a
        card for a game ESPN priced.
        """
        lines = parse_core_odds(self._spread_total_only())
        by_view = {line["viewType"]: line for line in lines}
        self.assertIn("Total", by_view)
        self.assertIn("Spread", by_view)
        # extract_total_line parses the "o165.5 (-110)" spelling SBR uses.
        self.assertIn("165.5", str(by_view["Total"]["currentLine"]["over"]))
        # ESPN quotes the spread from the home side; away is its mirror.
        self.assertIn("-4.5", str(by_view["Spread"]["currentLine"]["home"]))
        self.assertIn("+4.5", str(by_view["Spread"]["currentLine"]["away"]))

    def test_captured_markets_parse_back_out(self) -> None:
        """Round-trip through the model's own extractors, not just the shape."""
        from mlb_predictions import extract_spread_line, extract_total_line

        lines = parse_core_odds(self._spread_total_only())
        self.assertEqual(extract_total_line(lines), 165.5)
        self.assertEqual(extract_spread_line(lines), -4.5)

    def test_prediction_sites_are_skipped(self) -> None:
        """Folding a model's own output back in as "the market" is circular."""
        items = [book(name, -300, 250) for name in ("numberFire", "TeamRankings", "Consensus")]
        self.assertEqual(parse_core_odds({"items": items}), [])

    def test_legacy_decimal_provider_is_skipped(self) -> None:
        payload = {"items": [book("Legacy", -140, 120, provider_id="2000")]}
        self.assertEqual(parse_core_odds(payload), [])

    def test_unnamed_provider_is_skipped(self) -> None:
        payload = {"items": [{"provider": {}, "homeTeamOdds": {"moneyLine": -140},
                              "awayTeamOdds": {"moneyLine": 120}}]}
        self.assertEqual(parse_core_odds(payload), [])

    def test_zero_odds_treated_as_missing(self) -> None:
        """ESPN uses 0 as a null, and 0 is not a valid American price."""
        self.assertEqual(parse_core_odds({"items": [book("X", 0, 0)]}), [])

    def test_even_money_words(self) -> None:
        payload = {"items": [{"provider": {"id": "1", "name": "X"},
                              "homeTeamOdds": {"moneyLine": "EVEN"},
                              "awayTeamOdds": {"moneyLine": "-120"}}]}
        self.assertEqual(parse_core_odds(payload)[0]["currentLine"], {"home": 100, "away": -120})

    def test_multiple_books_all_survive(self) -> None:
        """Several books is what gives the de-vig a consensus to average."""
        items = [book("ESPN BET", -140, 120), book("DraftKings", -145, 125),
                 book("FanDuel", -138, 118)]
        self.assertEqual(len(parse_core_odds({"items": items})), 3)

    def test_garbage_payloads_are_empty_not_fatal(self) -> None:
        for payload in ({}, {"items": None}, {"items": []}, {"items": ["x", 1, None]}, None):
            self.assertEqual(parse_core_odds(payload), [])


class ModelCompatibilityTests(unittest.TestCase):
    """The output has to be consumable by the existing probability path."""

    def test_lines_satisfy_has_moneyline_lines(self) -> None:
        lines = parse_core_odds({"items": [book("ESPN BET", -140, 120)]})
        self.assertTrue(has_moneyline_lines(lines))

    def test_lines_produce_implied_probabilities(self) -> None:
        lines = parse_core_odds({"items": [book("ESPN BET", -140, 120),
                                           book("DraftKings", -145, 125)]})
        implied = compute_implied_probabilities(lines)
        self.assertTrue(implied["available"])
        self.assertEqual(implied["booksUsed"], 2)
        consensus = implied["consensus"]
        # -140/+120 is a home favourite; de-vigged home must exceed away.
        self.assertGreater(consensus["home"], consensus["away"])
        self.assertAlmostEqual(consensus["home"] + consensus["away"], 1.0, places=6)

    def test_three_way_probabilities_sum_to_one(self) -> None:
        lines = parse_core_odds({"items": [book("Bet365", 150, 200, draw=230)]})
        consensus = compute_implied_probabilities(lines)["consensus"]
        total = consensus["home"] + consensus["away"] + consensus["draw"]
        self.assertAlmostEqual(total, 1.0, places=6)


class FetchTests(unittest.TestCase):
    def test_network_failure_returns_empty_never_raises(self) -> None:
        from sbr_client import SBRFetchError

        with patch("espn_odds.get_text", side_effect=SBRFetchError("503")):
            self.assertEqual(fetch_event_odds("wnba", "1"), [])

    def test_bad_json_returns_empty(self) -> None:
        with patch("espn_odds.get_text", return_value="not json"):
            self.assertEqual(fetch_event_odds("wnba", "1"), [])

    def test_success_path(self) -> None:
        payload = json.dumps({"items": [book("ESPN BET", -140, 120)]})
        with patch("espn_odds.get_text", return_value=payload):
            self.assertEqual(len(fetch_event_odds("wnba", "1")), 1)


class FillMissingTests(unittest.TestCase):
    def _game(self, event_id: str, lines: list | None = None) -> dict:
        return {"eventId": event_id, "awayTeam": "A", "homeTeam": "B", "lines": lines or []}

    def test_prices_an_unpriced_game(self) -> None:
        games = [self._game("1")]
        with patch("espn_odds.fetch_event_odds",
                   return_value=parse_core_odds({"items": [book("ESPN BET", -140, 120)]})):
            stats = fill_missing_moneylines(games, league="wnba")

        self.assertEqual(stats["priced"], 1)
        self.assertTrue(has_moneyline_lines(games[0]["lines"]))
        self.assertEqual(games[0]["oddsSource"], "espn-core")
        self.assertIn("MoneyLine", games[0]["viewTypes"])

    def test_already_priced_game_is_never_fetched(self) -> None:
        """Cost control: MLB is priced by SBR, so this must make no requests."""
        priced = [{"viewType": "MoneyLine", "sportsbook": "SBR",
                   "currentLine": {"homeOdds": -150, "awayOdds": 130}}]
        games = [self._game("1", priced)]
        with patch("espn_odds.fetch_event_odds", side_effect=AssertionError("must not fetch")):
            stats = fill_missing_moneylines(games, league="mlb")
        self.assertEqual(stats["considered"], 0)
        self.assertEqual(stats["fetched"], 0)

    def test_spread_only_lines_still_count_as_unpriced(self) -> None:
        """A game carrying SBR spreads but no moneyline is exactly the WNBA case."""
        spread_only = [{"viewType": "Spread", "sportsbook": "SBR",
                        "currentLine": {"home": "-4.5 (-110)"}}]
        games = [self._game("1", spread_only)]
        with patch("espn_odds.fetch_event_odds",
                   return_value=parse_core_odds({"items": [book("ESPN BET", -140, 120)]})):
            stats = fill_missing_moneylines(games, league="wnba")

        self.assertEqual(stats["priced"], 1)
        # The spread line survives alongside the new moneyline.
        self.assertEqual(len(games[0]["lines"]), 2)
        self.assertTrue(has_moneyline_lines(games[0]["lines"]))

    def test_no_odds_available_leaves_the_game_alone(self) -> None:
        games = [self._game("1")]
        with patch("espn_odds.fetch_event_odds", return_value=[]):
            stats = fill_missing_moneylines(games, league="wnba")
        self.assertEqual(stats["considered"], 1)
        self.assertEqual(stats["priced"], 0)
        self.assertEqual(games[0]["lines"], [])

    def test_game_without_event_id_is_skipped(self) -> None:
        games = [{"awayTeam": "A", "homeTeam": "B", "lines": []}]
        with patch("espn_odds.fetch_event_odds", side_effect=AssertionError("must not fetch")):
            stats = fill_missing_moneylines(games, league="wnba")
        self.assertEqual(stats["fetched"], 0)

    def test_max_events_caps_the_request_count(self) -> None:
        games = [self._game(str(i)) for i in range(10)]
        with patch("espn_odds.fetch_event_odds", return_value=[]):
            stats = fill_missing_moneylines(games, league="wnba", max_events=3)
        self.assertEqual(stats["considered"], 10)
        self.assertEqual(stats["fetched"], 3)

    def test_books_used_are_recorded(self) -> None:
        lines = parse_core_odds({"items": [book("ESPN BET", -140, 120),
                                           book("DraftKings", -145, 125)]})
        with patch("espn_odds.fetch_event_odds", return_value=lines):
            stats = fill_missing_moneylines([self._game("1")], league="afl")
        self.assertEqual(sorted(stats["books"]), ["DraftKings", "ESPN BET"])


class ProviderFilterTests(unittest.TestCase):
    def test_filter_list_is_lowercase(self) -> None:
        """Matching is done casefolded, so the constants must be too."""
        for name in NON_BOOK_PROVIDERS:
            self.assertEqual(name, name.casefold())


class CircuitBreakerTests(unittest.TestCase):
    """A league ESPN does not cover must not be retried forever.

    Measured on the live build of 2026-07-29: ESPN prices WNBA but returns
    nothing for AFL. Without a breaker every AFL game is re-requested on every
    build -- roughly 400 pointless calls a day for a permanent negative.
    """

    def _games(self, count: int) -> list[dict]:
        return [{"eventId": str(i), "awayTeam": "A", "homeTeam": "B", "lines": []}
                for i in range(count)]

    def test_gives_up_after_consecutive_empties(self) -> None:
        with patch("espn_odds.fetch_event_odds", return_value=[]):
            stats = fill_missing_moneylines(self._games(12), league="afl")
        self.assertEqual(stats["considered"], 12)
        self.assertEqual(stats["fetched"], 3)
        self.assertTrue(stats["gaveUp"])

    def test_a_hit_resets_the_counter(self) -> None:
        """A slate where only some games are posted yet must not trip it."""
        priced = parse_core_odds({"items": [book("DraftKings", -140, 120)]})
        # empty, empty, hit, empty, empty, hit ... never three in a row
        responses = [[], [], priced, [], [], priced, [], []]
        with patch("espn_odds.fetch_event_odds", side_effect=responses):
            stats = fill_missing_moneylines(self._games(8), league="wnba")
        self.assertFalse(stats["gaveUp"])
        self.assertEqual(stats["fetched"], 8)
        self.assertEqual(stats["priced"], 2)

    def test_a_fully_priced_league_never_trips_it(self) -> None:
        priced = parse_core_odds({"items": [book("DraftKings", -140, 120)]})
        with patch("espn_odds.fetch_event_odds", return_value=priced):
            stats = fill_missing_moneylines(self._games(6), league="wnba")
        self.assertFalse(stats["gaveUp"])
        self.assertEqual(stats["priced"], 6)


if __name__ == "__main__":
    unittest.main()
