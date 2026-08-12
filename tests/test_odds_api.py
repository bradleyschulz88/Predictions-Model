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
import tempfile
import time
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


def _ok(body: str, headers: dict[str, str] | None = None):
    """What get_text_with_headers hands back: the body, then the headers.

    Most tests here care only about the body, so the headers default to none at
    all -- which is also a real case, since an origin is free not to send them.
    """
    return body, dict(headers or {})


class ConfigurationTests(unittest.TestCase):
    """Without a key this must be inert, not broken."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def test_no_key_means_no_call_and_no_prices(self) -> None:
        games = [_game()]
        with patch.dict("os.environ", {}, clear=False) as _, \
             patch.object(odds_api.os, "environ", {}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            stats = odds_api.attach_odds_to_games(games, league="afl")
        self.assertFalse(stats["configured"])
        self.assertEqual(stats["priced"], 0)
        self.assertEqual(games[0]["lines"], [])

    def test_an_uncovered_league_never_spends_a_credit(self) -> None:
        """NBA, NFL, WNBA and EPL are already served for nothing.

        MLB is no longer in this list, but only for one market -- see
        SpreadFillTests. It stays in the moneyline case below, because
        SportsBookReview supplies that for free.
        """
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            for league in ("nba", "nfl", "wnba", "epl"):
                stats = odds_api.attach_odds_to_games([_game()], league=league)
                self.assertEqual(stats["priced"], 0, msg=league)

    def test_baseball_never_spends_a_credit_on_a_moneyline(self) -> None:
        """MLB is configured for its runline and asks for `spreads` alone.

        A call from the moneyline path could not return an h2h price even if it
        succeeded, so it would spend a credit and attach nothing. Gated on the
        market rather than on the league, or adding MLB to LEAGUE_ODDS would
        have quietly opened this path too.
        """
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            stats = odds_api.attach_odds_to_games([_game()], league="mlb")
        self.assertEqual(stats["priced"], 0)


class BudgetTests(unittest.TestCase):
    """500 credits a month against ~1,440 builds is the whole design problem."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def test_a_game_already_priced_elsewhere_is_never_fetched_for(self) -> None:
        priced = _game()
        priced["lines"] = [{"viewType": "MoneyLine", "currentLine": {"home": "-120", "away": "+100"}}]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            stats = odds_api.attach_odds_to_games([priced], league="afl")
        self.assertEqual(stats["considered"], 0)

    def test_repeat_builds_inside_the_window_share_one_call(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))) as fetch:
            for _ in range(5):
                odds_api.attach_odds_to_games([_game()], league="afl")
        self.assertEqual(fetch.call_count, 1, "each build must not cost its own credits")

    def test_the_cache_expires_so_prices_do_not_go_stale_forever(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))) as fetch:
            odds_api.fetch_league_odds("afl", now=0.0)
            odds_api.fetch_league_odds("afl", now=odds_api.CACHE_TTL_SECONDS + 1)
        self.assertEqual(fetch.call_count, 2)

    def test_a_failure_is_cached_too(self) -> None:
        """A key out of credits fails every call; retrying each build helps nobody."""
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=OSError("quota")) as fetch:
            for _ in range(4):
                odds_api.attach_odds_to_games([_game()], league="afl")
        self.assertEqual(fetch.call_count, 1)

    def test_a_failure_never_raises(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=OSError("down")):
            self.assertEqual(odds_api.fetch_league_odds("afl"), [])


class LineShapeTests(unittest.TestCase):
    """The output has to be the shape the model already consumes."""

    def setUp(self) -> None:
        odds_api.clear_cache()

    def _lines(self):
        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))):
            odds_api.attach_odds_to_games(games, league="afl")
        return {line["viewType"]: line["currentLine"] for line in games[0]["lines"]}

    def test_decimal_prices_become_american(self) -> None:
        self.assertEqual(self._lines()["MoneyLine"], {"home": "-118", "away": "-105"})

    def test_side_market_prices_use_the_parenthetical_form_the_parsers_read(self) -> None:
        """extract_total_price and extract_spread_price look for "(-110)"."""
        from mlb_predictions import extract_spread_price, extract_total_price

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))):
            odds_api.attach_odds_to_games(games, league="afl")
        lines = games[0]["lines"]
        self.assertEqual(extract_total_price(lines, "over"), -110)
        self.assertEqual(extract_spread_price(lines, "home"), -111)

    def test_the_spread_line_is_readable_as_a_number(self) -> None:
        from mlb_predictions import extract_spread_line, extract_total_line

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))):
            odds_api.attach_odds_to_games(games, league="afl")
        self.assertAlmostEqual(extract_spread_line(games[0]["lines"]), -5.5)
        self.assertAlmostEqual(extract_total_line(games[0]["lines"]), 165.5)

    def test_it_reaches_a_real_expected_value(self) -> None:
        """The point of all of it: AFL picks could not be valued at all."""
        from mlb_predictions import apply_predictions

        games = [_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(_payload()))):
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
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(other))):
            stats = odds_api.attach_odds_to_games(games, league="afl")
        self.assertEqual(stats["priced"], 0)
        self.assertEqual(games[0]["lines"], [])


class SpreadFillTests(unittest.TestCase):
    """MLB runlines, the one market with no free price anywhere.

    SportsBookReview gives baseball a moneyline and ESPN core now prices 85 of
    86 totals, but ESPN publishes the runline handicap with no juice on it --
    {"home": "-1.5"} where the same endpoint returns WNBA
    {"home": "-6.5 (-112)"}. Measured 2026-08-05: 0 of 62 graded runlines had
    a price, so that market could never be valued or ranked.

    The trap is the entry point. attach_odds_to_games only looks at games with
    no moneyline at all, which is right for AFL and useless here, because every
    MLB game has one.
    """

    HOME, AWAY = "Houston Astros", "Toronto Blue Jays"

    def setUp(self) -> None:
        odds_api.clear_cache()

    def _mlb_payload(self):
        return [{
            "id": "m1", "home_team": self.HOME, "away_team": self.AWAY,
            "bookmakers": [{"key": "dk", "title": "DraftKings", "markets": [
                {"key": "spreads", "outcomes": [
                    {"name": self.HOME, "price": 2.05, "point": -1.5},
                    {"name": self.AWAY, "price": 1.8, "point": 1.5}]},
            ]}],
        }]

    def _mlb_game(self):
        """As the board has it: moneyline priced, runline bare."""
        return {
            "eventId": "401816410", "league": "mlb",
            "homeTeam": self.HOME, "awayTeam": self.AWAY,
            "lines": [
                {"sportsbook": "SBR", "viewType": "MoneyLine",
                 "currentLine": {"home": "-208", "away": "+195"}},
                {"sportsbook": "ESPN BET", "viewType": "Spread",
                 "currentLine": {"home": "-1.5", "away": "+1.5"}},
            ],
        }

    def test_a_priced_moneyline_does_not_hide_an_unpriced_runline(self) -> None:
        """The bug this exists for: the game looks covered and is not."""
        games = [self._mlb_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(self._mlb_payload()))):
            stats = odds_api.fill_missing_spread_prices(games, league="mlb")
        self.assertEqual(stats["considered"], 1)
        self.assertEqual(stats["priced"], 1)

    def test_the_runline_price_is_readable_by_the_extractor(self) -> None:
        """End to end: what predict_runline actually calls has to return a number."""
        from mlb_predictions import extract_spread_price

        games = [self._mlb_game()]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(self._mlb_payload()))):
            odds_api.fill_missing_spread_prices(games, league="mlb")
        self.assertEqual(extract_spread_price(games[0]["lines"], "home"), 105)
        self.assertEqual(extract_spread_price(games[0]["lines"], "away"), -125)

    def test_only_spread_rows_are_merged(self) -> None:
        """A second moneyline would land in the consensus twice."""
        games = [self._mlb_game()]
        payload = self._mlb_payload()
        payload[0]["bookmakers"][0]["markets"].append(
            {"key": "h2h", "outcomes": [{"name": self.HOME, "price": 1.5},
                                        {"name": self.AWAY, "price": 2.6}]})
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(payload))):
            odds_api.fill_missing_spread_prices(games, league="mlb")
        views = [line["viewType"] for line in games[0]["lines"]]
        self.assertEqual(views.count("MoneyLine"), 1)

    def test_an_already_priced_spread_is_left_alone(self) -> None:
        """No credit for a game a free source already covered."""
        game = self._mlb_game()
        game["lines"][1]["currentLine"] = {"home": "-1.5 (+105)", "away": "+1.5 (-125)"}
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            stats = odds_api.fill_missing_spread_prices([game], league="mlb")
        self.assertEqual(stats["considered"], 0)
        self.assertEqual(stats["priced"], 0)

    def test_no_key_means_no_call(self) -> None:
        with patch.object(odds_api.os, "environ", {}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            stats = odds_api.fill_missing_spread_prices([self._mlb_game()], league="mlb")
        self.assertFalse(stats["configured"])
        self.assertEqual(stats["priced"], 0)

    def test_a_league_without_a_spreads_market_never_calls(self) -> None:
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            for league in ("nba", "nfl", "wnba", "epl"):
                stats = odds_api.fill_missing_spread_prices([self._mlb_game()], league=league)
                self.assertEqual(stats["priced"], 0, msg=league)


class CreditCostTests(unittest.TestCase):
    """A credit is charged per market per region, so the ask is the price."""

    def test_baseball_asks_for_one_market_not_three(self) -> None:
        """h2h and totals both have free sources; paying for them is waste."""
        self.assertEqual(odds_api.LEAGUE_ODDS["mlb"].markets, "spreads")
        self.assertEqual(odds_api.LEAGUE_ODDS["mlb"].credits_per_call, 1)

    def test_afl_still_asks_for_everything_because_it_has_nothing(self) -> None:
        self.assertEqual(odds_api.LEAGUE_ODDS["afl"].credits_per_call, 3)

    def test_the_whole_board_fits_the_free_tier(self) -> None:
        """Four calls a day each on a six-hour cache, against 500 a month."""
        monthly = sum(
            config.credits_per_call * 4 * 30 for config in odds_api.LEAGUE_ODDS.values()
        )
        self.assertLess(monthly, 500, f"{monthly} credits a month exceeds the free tier")

    def test_baseball_uses_us_books(self) -> None:
        """`au` would be an Australian shop guessing at an MLB runline."""
        self.assertEqual(odds_api.LEAGUE_ODDS["mlb"].regions, "us")


class CachePersistenceTests(unittest.TestCase):
    """The six-hour TTL is worth nothing if the cache dies with the process.

    CI starts a fresh interpreter every thirty minutes, so an in-process dict
    is empty on arrival every time and every build pays again. For the ESPN
    pass that was rudeness; here it is money -- MLB alone would spend a credit
    on each of ~48 builds a day, about 1,440 a month against an allowance of
    500, so the whole free tier would be gone inside a week.
    """

    def setUp(self) -> None:
        odds_api.clear_cache()
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name) / "nested" / "odds-api.json"
        patcher = patch.dict(
            odds_api.os.environ, {odds_api.CACHE_FILE_ENV: str(self.path)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _seed(self, age_seconds: float = 0.0) -> None:
        odds_api._CACHE["baseball_mlb"] = (time.time() - age_seconds, [{"id": "m1"}])

    def test_a_saved_slate_comes_back_next_build(self) -> None:
        self._seed()
        self.assertEqual(odds_api.save_cache(), 1)
        odds_api.clear_cache()
        self.assertEqual(odds_api.load_cache(), 1)
        self.assertEqual(odds_api._CACHE["baseball_mlb"][1], [{"id": "m1"}])

    def test_a_restored_slate_spends_no_credit(self) -> None:
        """The whole point. If this fetches, the quota is being burned."""
        self._seed()
        odds_api.save_cache()
        odds_api.clear_cache()
        odds_api.load_cache()
        with patch.object(odds_api.os, "environ",
                          {"ODDS_API_KEY": "k", odds_api.CACHE_FILE_ENV: str(self.path)}), \
             patch.object(odds_api, "get_text_with_headers", side_effect=AssertionError("must not call")):
            events = odds_api.fetch_league_odds("mlb")
        self.assertEqual(events, [{"id": "m1"}])

    def test_an_expired_slate_is_dropped_on_the_way_in(self) -> None:
        """A stale file must not pin an old price onto a live board."""
        stale = time.time() - odds_api.CACHE_TTL_SECONDS - 60
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"entries": {"baseball_mlb": {"fetchedAt": stale, "events": [{"id": "m1"}]}}}
        ), encoding="utf-8")
        self.assertEqual(odds_api.load_cache(), 0)
        self.assertEqual(odds_api._CACHE, {})

    def test_an_expired_slate_is_not_written_out(self) -> None:
        self._seed(age_seconds=odds_api.CACHE_TTL_SECONDS + 60)
        self.assertEqual(odds_api.save_cache(), 0)

    def test_a_missing_or_corrupt_file_is_not_an_error(self) -> None:
        self.assertEqual(odds_api.load_cache(), 0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(odds_api.load_cache(), 0)

    def test_no_configured_path_disables_persistence_silently(self) -> None:
        self._seed()
        with patch.dict(odds_api.os.environ, {odds_api.CACHE_FILE_ENV: ""}):
            self.assertEqual(odds_api.save_cache(), 0)
            self.assertEqual(odds_api.load_cache(), 0)

    def test_valid_json_of_the_wrong_shape_is_not_an_error(self) -> None:
        """A bare list or null parses fine and then raises on .get.

        Caught by fuzzing the loader: only OSError and JSONDecodeError were
        handled, so `[]` or `null` in the cache file threw AttributeError out
        of build start-up -- the exact outcome this function exists to avoid.
        The file comes back from a shared actions/cache key, so its contents
        are not something this code gets to assume.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for payload in ("[]", "null", '"a string"', "123", "true"):
            self.path.write_text(payload, encoding="utf-8")
            self.assertEqual(odds_api.load_cache(), 0, payload)


class FixtureOrientationTests(unittest.TestCase):
    """A match must not attach a fixture's prices with the sides swapped.

    _match_event scores home-against-home and away-against-away, which reads
    like it settles orientation. It does not when both clubs share a city:
    "New York Yankees" against "New York Mets" scores 0.667 on two of three
    matching words, so an event with the clubs the wrong way round clears the
    0.6 bar just as the right one does.

    The consequence is not a missing price, it is a wrong one -- the home side
    handed the away side's handicap, turning a -1.5 favourite into a +1.5
    underdog on the market whose entire purpose is to be valued. The API
    returns a slate spanning several days, so a later reversed fixture between
    the same two clubs is an ordinary thing to find in the response.

    A per-side floor does not fix it: 0.667 clears any floor loose enough to
    tolerate the naming differences this matcher exists for. Orientation has
    to be decided against the mirror image instead.
    """

    def _event(self, home, away):
        return {"id": f"{home}|{away}", "home_team": home, "away_team": away}

    SAME_CITY = [
        ("New York Yankees", "New York Mets"),
        ("Los Angeles Dodgers", "Los Angeles Angels"),
        ("Chicago White Sox", "Chicago Cubs"),
    ]

    def test_a_reversed_fixture_is_not_matched(self) -> None:
        for home, away in self.SAME_CITY:
            got = odds_api._match_event(
                {"homeTeam": home, "awayTeam": away}, [self._event(away, home)]
            )
            self.assertIsNone(got, f"{away} @ {home} matched its own mirror image")

    def test_the_right_way_round_still_matches(self) -> None:
        for home, away in self.SAME_CITY:
            got = odds_api._match_event(
                {"homeTeam": home, "awayTeam": away}, [self._event(home, away)]
            )
            self.assertIsNotNone(got, f"{away} @ {home} no longer matches itself")

    def test_the_right_event_wins_when_both_orientations_are_present(self) -> None:
        """Both appear in one response when two clubs meet again days later."""
        home, away = "New York Yankees", "New York Mets"
        got = odds_api._match_event(
            {"homeTeam": home, "awayTeam": away},
            [self._event(away, home), self._event(home, away)],
        )
        self.assertEqual(got["id"], f"{home}|{away}")

    def test_loose_club_naming_still_matches(self) -> None:
        """The fuzziness this matcher exists for must survive the fix."""
        got = odds_api._match_event(
            {"homeTeam": "Carlton Blues", "awayTeam": "Collingwood Magpies"},
            [self._event("Carlton", "Collingwood")],
        )
        self.assertIsNotNone(got)

    def test_a_reversed_match_cannot_reach_the_lines(self) -> None:
        """End to end: no prices attached, rather than inverted ones."""
        game = {
            "eventId": "1", "league": "mlb",
            "homeTeam": "New York Yankees", "awayTeam": "New York Mets",
            "lines": [{"sportsbook": "SBR", "viewType": "MoneyLine",
                       "currentLine": {"home": "-150", "away": "+130"}}],
        }
        reversed_event = [{
            "id": "r", "home_team": "New York Mets", "away_team": "New York Yankees",
            "bookmakers": [{"key": "dk", "title": "DraftKings", "markets": [
                {"key": "spreads", "outcomes": [
                    {"name": "New York Mets", "price": 2.05, "point": -1.5},
                    {"name": "New York Yankees", "price": 1.8, "point": 1.5}]}]}],
        }]
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers", return_value=_ok(json.dumps(reversed_event))):
            stats = odds_api.fill_missing_spread_prices([game], league="mlb")
        self.assertEqual(stats["priced"], 0)
        self.assertEqual([line["viewType"] for line in game["lines"]], ["MoneyLine"])


class QuotaReportingTests(unittest.TestCase):
    """The credit balance has to leave the response and reach the log.

    It did not, for the whole time this provider had been live. The module
    declared `_QUOTA`, exposed `quota_status()`, and `scripts/build_pages_data`
    printed `Odds API quota: {...}` whenever that came back non-empty -- but
    nothing ever wrote to the dict, because `get_text` returns a body and drops
    the response object that carries the headers. So the line never printed
    once, and the 11 Aug 2026 build spent credits on 7 slates with no record of
    what was left. Verified against that run's log: no quota line appears.

    That matters more here than it would elsewhere. The free plan is 500 credits
    a month and the failure mode of running out is silent -- every call fails,
    the failure is cached, and prices simply stop appearing.
    """

    def setUp(self) -> None:
        odds_api.clear_cache()

    def _fetch(self, headers):
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers",
                          return_value=_ok(json.dumps(_payload()), headers)):
            odds_api.fetch_league_odds("afl")
        return odds_api.quota_status()

    def test_the_balance_is_read_off_the_response(self) -> None:
        quota = self._fetch({"x-requests-remaining": "472", "x-requests-used": "28"})
        self.assertEqual(quota["remaining"], 472)
        self.assertEqual(quota["used"], 28)

    def test_the_cost_of_the_last_call_is_recorded(self) -> None:
        """Three for AFL, one for MLB -- worth seeing rather than assuming."""
        self.assertEqual(self._fetch({"x-requests-last": "3"})["lastCallCost"], 3)

    def test_headers_are_matched_whatever_case_they_arrive_in(self) -> None:
        self.assertEqual(self._fetch({"X-Requests-Remaining": "5"}).get("remaining"), 5)

    def test_numbers_are_numbers_and_nonsense_is_dropped(self) -> None:
        quota = self._fetch({"x-requests-remaining": "not a number", "x-requests-used": "9"})
        self.assertNotIn("remaining", quota)
        self.assertEqual(quota["used"], 9)

    def test_nothing_fetched_means_no_claim_about_the_budget(self) -> None:
        """Empty must read as "no call made", never as "no credits left"."""
        self.assertEqual(odds_api.quota_status(), {})

    def test_a_cache_hit_leaves_the_last_known_balance_alone(self) -> None:
        """A cached slate costs nothing, so it reports nothing new."""
        self._fetch({"x-requests-remaining": "400"})
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers",
                          side_effect=AssertionError("must not call")):
            odds_api.fetch_league_odds("afl")
        self.assertEqual(odds_api.quota_status()["remaining"], 400)

    def test_a_credit_spent_on_an_unreadable_reply_is_still_counted(self) -> None:
        """The charge lands when the request does, not when the JSON parses."""
        with patch.object(odds_api.os, "environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers",
                          return_value=_ok("<html>rate limited</html>",
                                           {"x-requests-remaining": "0"})):
            self.assertEqual(odds_api.fetch_league_odds("afl"), [])
        self.assertEqual(odds_api.quota_status()["remaining"], 0)

    def test_the_build_prints_what_it_reads(self) -> None:
        """Capturing the number is only half of it; it has to be visible."""
        source = (ROOT / "scripts" / "build_pages_data.py").read_text(encoding="utf-8")
        self.assertIn("Odds API quota", source)


class QuotaSurvivesBetweenBuildsTests(unittest.TestCase):
    """The balance has to outlive the interpreter, or it is invisible in practice.

    Reading the headers was necessary and not sufficient. A build that answers
    entirely from the disk cache makes no call, so it has no headers and can say
    nothing about the budget -- and because the cache is doing its job, that is
    most builds. Six sampled builds on 11 Aug between 03:01Z and 23:53Z were all
    cache hits, so the figure still never appeared in a log.

    So the balance is written into the cache file alongside the slates and read
    back on the next build, stamped with when it was actually taken.
    """

    def setUp(self) -> None:
        odds_api.clear_cache()
        self.dir = tempfile.mkdtemp()
        self.path = str(Path(self.dir) / "odds-api.json")
        patcher = patch.dict("os.environ", {odds_api.CACHE_FILE_ENV: self.path})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(odds_api.clear_cache)

    def _fetch_with(self, headers):
        # patch.dict rather than replacing os.environ wholesale: the cache path
        # is an environment variable too, and swapping the mapping would take
        # it with it -- leaving the file this whole class is about unwritten.
        with patch.dict("os.environ", {"ODDS_API_KEY": "k"}), \
             patch.object(odds_api, "get_text_with_headers",
                          return_value=_ok(json.dumps(_payload()), headers)):
            odds_api.fetch_league_odds("afl")

    def test_the_balance_is_written_to_disk(self) -> None:
        self._fetch_with({"x-requests-remaining": "463"})
        odds_api.save_cache()
        written = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.assertEqual(written["quota"]["remaining"], 463)

    def test_a_later_build_reads_it_back_without_spending_a_credit(self) -> None:
        self._fetch_with({"x-requests-remaining": "463", "x-requests-used": "37"})
        odds_api.save_cache()

        odds_api.clear_cache()  # a fresh interpreter, as CI gives every build
        self.assertEqual(odds_api.quota_status(), {}, "precondition: starts empty")
        odds_api.load_cache()
        self.assertEqual(odds_api.quota_status()["remaining"], 463)
        self.assertEqual(odds_api.quota_status()["used"], 37)

    def test_the_carried_balance_says_when_it_was_taken(self) -> None:
        """So a figure from six hours ago cannot pass as a live one."""
        self._fetch_with({"x-requests-remaining": "463"})
        stamped = odds_api.quota_status()["asOf"]
        odds_api.save_cache()
        odds_api.clear_cache()
        odds_api.load_cache()
        self.assertEqual(odds_api.quota_status()["asOf"], stamped)

    def test_the_stamp_only_moves_when_a_call_actually_reports(self) -> None:
        self._fetch_with({"x-requests-remaining": "463"})
        first = odds_api.quota_status()["asOf"]
        # A response carrying no quota headers must not restamp the old figure.
        odds_api._record_quota({"content-type": "application/json"})
        self.assertEqual(odds_api.quota_status()["asOf"], first)
        self.assertEqual(odds_api.quota_status()["remaining"], 463)

    def test_expired_slates_do_not_take_the_balance_with_them(self) -> None:
        """Prices go stale in six hours. A credit count does not."""
        Path(self.path).write_text(json.dumps({
            "entries": {"aussierules_afl": {
                "fetchedAt": time.time() - odds_api.CACHE_TTL_SECONDS - 60,
                "events": [],
            }},
            "quota": {"remaining": 400, "asOf": int(time.time()) - 99999},
        }), encoding="utf-8")
        self.assertEqual(odds_api.load_cache(), 0, "the slate should have expired")
        self.assertEqual(odds_api.quota_status()["remaining"], 400)

    def test_a_junk_quota_block_is_ignored_rather_than_fatal(self) -> None:
        for junk in ("not a dict", None, {"remaining": "lots"}, {"remaining": True}):
            with self.subTest(junk=junk):
                odds_api.clear_cache()
                Path(self.path).write_text(
                    json.dumps({"entries": {}, "quota": junk}), encoding="utf-8"
                )
                odds_api.load_cache()
                self.assertNotIn("remaining", odds_api.quota_status())

    def test_the_build_distinguishes_unknown_from_exhausted(self) -> None:
        """Empty must never read as "no credits left"."""
        source = (ROOT / "scripts" / "build_pages_data.py").read_text(encoding="utf-8")
        self.assertIn("not known yet", source)


class ResponseHeaderTests(unittest.TestCase):
    """`get_text` had no way to hand a caller the headers, hence the bug above."""

    def test_the_plain_fetcher_still_returns_just_the_text(self) -> None:
        """Every other scraper in the project calls it and wants a string."""
        import sbr_client

        with patch.object(sbr_client, "get_text_with_headers", return_value=("body", {})):
            self.assertEqual(sbr_client.get_text("https://example.test"), "body")

    def test_header_names_are_lower_cased_for_the_caller(self) -> None:
        """So a caller never has to guess how an origin capitalised them."""
        import sbr_client

        class _Response:
            headers = {"X-Requests-Remaining": "12"}

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch.object(sbr_client.urllib.request, "urlopen", return_value=_Response()):
            _, headers = sbr_client.get_text_with_headers("https://example.test")
        self.assertEqual(headers, {"x-requests-remaining": "12"})


if __name__ == "__main__":
    unittest.main()
