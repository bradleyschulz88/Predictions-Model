"""The Odds API -- a price source for leagues nothing else covers.

AFL is why it exists: no SportsBookReview board and nothing in ESPN core, so
every AFL pick has been permanently unpriced -- no expected value, no stake,
and no way to rank it against a game in any other league.

The free plan allows 500 credits a month and charges per market per region, so
one h2h+spreads+totals call costs three. The site rebuilds every thirty
minutes. Most of what these tests pin is the budget discipline that keeps that
from exhausting a month's quota in a day.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_providers.odds_api as odds_api  # noqa: E402


def _payload(home="Carlton", away="Collingwood"):
    return [{
        "id": "e1", "home_team": home, "away_team": away,
        "bookmakers": [{"key": "sportsbet", "title": "SportsBet", "markets": [
            {"key": "h2h", "outcomes": [
                {"name": home, "price": 1.85}, {"name": away, "price": 1.95}]},
            {"key": "spreads", "outcomes": [
                {"name": home, "price": 1.9, "point": -5.5},
                {"name": away, "price": 1.9, "point": 5.5}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": 1.91, "point": 165.5},
                {"name": "Under", "price": 1.89, "point": 165.5}]},
        ]}],
    }]


def _game():
    return {"eventId": "1", "league": "afl", "homeTeam": "Carlton Blues",
            "awayTeam": "Collingwood Magpies", "lines": []}


class ConfigurationTests(unittest.TestCase):
    """Without a key this must be inert, not broken."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def test_no_key_means_no_call_and_no_prices(self) -> None:
        games = [_game()]
        with patch.dict("os.environ", {}, clear=False) as _, \
             patch.object(odds_api.os, "environ", {}), \
             patch.object(odds_api, "get_text", side_effect=AssertionError("must not call")):
            stats = odds_api.attach_odds_to_games(games, league="afl")
        self.assertFalse(stats["configured"])
        self.assertEqual(stats["priced"], 0)
        self.assertEqual(games[0]["lines"], [])

    def test_an_uncovered_league_never_spends_a_credit(self) -> None:
        """MLB, NBA, NFL, WNBA and EPL are already served for nothing."""
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", side_effect=AssertionError("must not call")):
            for league in ("mlb", "nba", "nfl", "wnba", "epl"):
                stats = odds_api.attach_odds_to_games([_game()], league=league)
                self.assertEqual(stats["priced"], 0, msg=league)


class BudgetTests(unittest.TestCase):
    """500 credits a month against ~1,440 builds is the whole design problem."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def test_a_game_already_priced_elsewhere_is_never_fetched_for(self) -> None:
        priced = _game()
        priced["lines"] = [{"viewType": "MoneyLine", "currentLine": {"home": "-120", "away": "+100"}}]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", side_effect=AssertionError("must not call")):
            stats = odds_api.attach_odds_to_games([priced], league="afl")
        self.assertEqual(stats["considered"], 0)

    def test_repeat_builds_inside_the_window_share_one_call(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())) as fetch:
            for _ in range(5):
                odds_api.attach_odds_to_games([_game()], league="afl")
        self.assertEqual(fetch.call_count, 1, "each build must not cost its own credits")

    def test_the_cache_expires_so_prices_do_not_go_stale_forever(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())) as fetch:
            odds_api.fetch_league_odds("afl", now=0.0)
            odds_api.fetch_league_odds("afl", now=odds_api.CACHE_TTL_SECONDS + 1)
        self.assertEqual(fetch.call_count, 2)

    def test_a_failure_is_cached_too(self) -> None:
        """A key out of credits fails every call; retrying each build helps nobody."""
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", side_effect=OSError("quota")) as fetch:
            for _ in range(4):
                odds_api.attach_odds_to_games([_game()], league="afl")
        self.assertEqual(fetch.call_count, 1)

    def test_a_failure_never_raises(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", side_effect=OSError("down")):
            self.assertEqual(odds_api.fetch_league_odds("afl"), [])


class LineShapeTests(unittest.TestCase):
    """The output has to be the shape the model already consumes."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def _lines(self):
        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())):
            odds_api.attach_odds_to_games(games, league="afl")
        return {line["viewType"]: line["currentLine"] for line in games[0]["lines"]}

    def test_decimal_prices_become_american(self) -> None:
        self.assertEqual(self._lines()["MoneyLine"], {"home": "-118", "away": "-105"})

    def test_side_market_prices_use_the_parenthetical_form_the_parsers_read(self) -> None:
        """extract_total_price and extract_spread_price look for "(-110)"."""
        from mlb_predictions import extract_spread_price, extract_total_price

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())):
            odds_api.attach_odds_to_games(games, league="afl")
        lines = games[0]["lines"]
        self.assertEqual(extract_total_price(lines, "over"), -110)
        self.assertEqual(extract_spread_price(lines, "home"), -111)

    def test_the_spread_line_is_readable_as_a_number(self) -> None:
        from mlb_predictions import extract_spread_line, extract_total_line

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())):
            odds_api.attach_odds_to_games(games, league="afl")
        self.assertAlmostEqual(extract_spread_line(games[0]["lines"]), -5.5)
        self.assertAlmostEqual(extract_total_line(games[0]["lines"]), 165.5)

    def test_it_reaches_a_real_expected_value(self) -> None:
        """The point of all of it: AFL picks could not be valued at all."""
        from mlb_predictions import apply_predictions

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(_payload())):
            odds_api.attach_odds_to_games(games, league="afl")
        apply_predictions(games)
        value = games[0]["prediction"].get("value")
        self.assertIsNotNone(value, "an AFL pick must now carry an expected value")
        self.assertIsNotNone(value.get("evPct"))


class MatchingTests(unittest.TestCase):
    """A wrong fixture's prices are worse than no prices."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def test_club_names_match_across_naming_styles(self) -> None:
        """A book writes "Carlton" where ESPN writes "Carlton Blues"."""
        self.assertEqual(odds_api._side_for("Carlton", "Carlton Blues", "Collingwood Magpies"), "home")
        self.assertEqual(odds_api._side_for("Collingwood", "Carlton Blues", "Collingwood Magpies"), "away")

    def test_an_ambiguous_name_attaches_to_neither_side(self) -> None:
        self.assertIsNone(odds_api._side_for("Someone Else", "Carlton Blues", "Collingwood Magpies"))

    def test_an_unrelated_fixture_is_not_matched(self) -> None:
        games = [_game()]
        other = _payload(home="Geelong", away="Hawthorn")
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text", return_value=json.dumps(other)):
            stats = odds_api.attach_odds_to_games(games, league="afl")
        self.assertEqual(stats["priced"], 0)
        self.assertEqual(games[0]["lines"], [])


if __name__ == "__main__":
    unittest.main()
