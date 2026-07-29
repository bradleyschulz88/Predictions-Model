"""Tests for ESPN schedule integration and dashboard API."""

from __future__ import annotations

import json
import threading
import unittest

from offline import OfflineTestCase
import urllib.request
from pathlib import Path

from dashboard_server import DashboardConfig, create_handler
from espn_client import parse_scoreboard
from http.server import ThreadingHTTPServer
from mlb_data import (
    build_dashboard_payload_from_espn_games,
    build_dashboard_payload_from_sbr,
    default_game_date,
    fetch_dashboard_data,
    load_page_props_from_file,
    strip_betting_lines_for_display,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ESPN_FIXTURE = FIXTURES / "espn_scoreboard_20260616.json"


class ESPNDataTests(OfflineTestCase):
    def test_parse_scoreboard_fixture(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            scoreboard = json.load(handle)
        games = parse_scoreboard(scoreboard, league="mlb")
        self.assertEqual(len(games), 15)
        self.assertEqual(games[0]["awayTeam"], "Miami Marlins")
        self.assertEqual(games[0]["homeTeam"], "Philadelphia Phillies")

    def test_fetch_dashboard_data_from_espn_fixture(self) -> None:
        payload = fetch_dashboard_data(fixture=ESPN_FIXTURE, include_odds=False, league="mlb")
        self.assertEqual(payload["source"], "espn")
        self.assertEqual(payload["gameCount"], 15)

    def test_strip_betting_lines_for_display(self) -> None:
        payload = {
            "sportsbookCount": 2,
            "sportsbooks": ["BookA"],
            "games": [{"eventId": "1", "lines": [{"viewType": "MoneyLine"}], "oddsSource": "espn"}],
        }
        cleaned = strip_betting_lines_for_display(payload)
        self.assertNotIn("sportsbookCount", cleaned)
        self.assertNotIn("lines", cleaned["games"][0])
        self.assertNotIn("oddsSource", cleaned["games"][0])

    def test_default_game_date_uses_league_timezone(self) -> None:
        self.assertRegex(default_game_date("mlb"), r"^\d{4}-\d{2}-\d{2}$")
        self.assertRegex(default_game_date("afl"), r"^\d{4}-\d{2}-\d{2}$")


AFL_FIXTURE = FIXTURES / "espn_afl_scoreboard_20260614.json"


class MultiSportTests(unittest.TestCase):
    def test_parse_afl_scoreboard(self) -> None:
        if not AFL_FIXTURE.is_file():
            self.skipTest("AFL fixture not downloaded")
        with open(AFL_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="afl")
        self.assertGreater(len(games), 0)
        self.assertEqual(games[0]["league"], "afl")

    def test_default_game_date_epl_is_today(self) -> None:
        self.assertRegex(default_game_date("epl"), r"^\d{4}-\d{2}-\d{2}$")

    def test_retired_league_is_rejected(self) -> None:
        """A league that is no longer configured must fail loudly, not silently
        fetch an empty schedule under a default config."""
        with self.assertRaises(ValueError):
            default_game_date("worldcup")


class DashboardDataTests(OfflineTestCase):
    def test_build_dashboard_payload_from_sbr_fixture(self) -> None:
        page_props = load_page_props_from_file(FIXTURES / "odds_page.json")
        payload = build_dashboard_payload_from_sbr(page_props, url="fixture:test")
        self.assertEqual(payload["gameCount"], 2)
        teams = {game["awayTeam"] for game in payload["games"]}
        self.assertIn("New York Yankees", teams)
        self.assertIn("prediction", payload["games"][0])

    def test_build_dashboard_payload_from_espn_games(self) -> None:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="mlb")
        payload = build_dashboard_payload_from_espn_games(games, url="fixture:espn")
        self.assertEqual(payload["gameCount"], 15)
        self.assertEqual(payload["source"], "espn")


class DashboardServerTests(OfflineTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = DashboardConfig(fixture=str(ESPN_FIXTURE), include_odds=False, source="espn")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(config))
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _get_json(self, path: str, *, timeout: float = 30) -> dict:
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}", timeout=timeout) as response:
            return json.load(response)

    def test_health_endpoint(self) -> None:
        payload = self._get_json("/api/health")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["source"], "espn")

    def test_games_endpoint(self) -> None:
        payload = self._get_json("/api/games?date=2026-06-16")
        self.assertEqual(payload["gameCount"], 15)
        self.assertEqual(payload["source"], "espn")

    def test_index_page(self) -> None:
        with urllib.request.urlopen(f"http://{self.host}:{self.port}/", timeout=5) as response:
            html = response.read().decode("utf-8")
        self.assertIn("Sports Predictions Dashboard", html)


if __name__ == "__main__":
    unittest.main()


class TeamAbbreviationTests(OfflineTestCase):
    """The board shows abbreviations, so a wrong one is what the user sees.

    Deriving them from the words produced "AB" for Atlanta Braves and "BRS" for
    Boston Red Sox. ESPN publishes the official abbreviation next to the name,
    so capture that instead of guessing.
    """

    def _games(self) -> list[dict]:
        with open(ESPN_FIXTURE, encoding="utf-8") as handle:
            return parse_scoreboard(json.load(handle), league="mlb")

    def test_scoreboard_captures_official_abbreviations(self) -> None:
        game = self._games()[0]
        self.assertEqual(game["awayAbbr"], "MIA")
        self.assertEqual(game["homeAbbr"], "PHI")

    def test_every_game_carries_both_abbreviations(self) -> None:
        for game in self._games():
            self.assertTrue(game.get("awayAbbr"), game.get("matchup"))
            self.assertTrue(game.get("homeAbbr"), game.get("matchup"))

    def test_abbreviations_are_not_word_initials(self) -> None:
        """The exact regression: initials would render these as AB and BRS."""
        for game in self._games():
            for side in ("away", "home"):
                name = game[f"{side}Team"]
                initials = "".join(word[0] for word in name.split())[:4].upper()
                if len(name.split()) == 2:
                    self.assertNotEqual(
                        game[f"{side}Abbr"], initials, f"{name} still abbreviating to initials"
                    )


class DashboardAbbrevFallbackTests(OfflineTestCase):
    """The JS fallback only fires for stored records with no captured abbrev."""

    def _app_js(self) -> str:
        return (Path(__file__).resolve().parents[1] / "dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

    def test_explicit_abbreviation_wins(self) -> None:
        app_js = self._app_js()
        self.assertIn("function teamAbbrev(name, explicit)", app_js)
        self.assertIn("if (given) return given.toUpperCase();", app_js)

    def test_two_word_names_use_the_city_not_initials(self) -> None:
        """"Atlanta Braves" must fall back to ATL, never AB."""
        app_js = self._app_js()
        self.assertIn("if (words.length === 2) return words[0].slice(0, 3).toUpperCase();", app_js)

    def test_call_sites_pass_the_captured_abbreviation(self) -> None:
        app_js = self._app_js()
        self.assertNotIn("teamAbbrev(game.homeTeam)", app_js)
        self.assertNotIn("teamAbbrev(game.awayTeam)", app_js)
        self.assertIn("teamAbbrev(game.homeTeam, game.homeAbbr)", app_js)
        self.assertIn("teamAbbrev(game.awayTeam, game.awayAbbr)", app_js)


class DoubleheaderDisplayTests(OfflineTestCase):
    """Doubleheaders are real games, not duplicates.

    Nine pairs in the logged history, every one with two different final scores.
    The model handles them correctly; the board collapsed both to the same line
    with the start time hidden inside <details>, so they read as a duplicate.
    """

    def _app_js(self) -> str:
        return (Path(__file__).resolve().parents[1] / "dashboard" / "app.js").read_text(
            encoding="utf-8"
        )

    def test_annotation_runs_before_the_cards_render(self) -> None:
        app_js = self._app_js()
        self.assertIn("function annotateDoubleheaders(games)", app_js)
        self.assertIn("annotateDoubleheaders(visible);", app_js)

    def test_badge_reaches_every_scoreboard_return_path(self) -> None:
        """Three paths render the matchup line -- scored, with form, plain."""
        app_js = self._app_js()
        start = app_js.index("function renderScoreboardTeams(game)")
        body = app_js[start : app_js.index("\n}", start)]
        self.assertEqual(body.count("${dh}"), 3, "a return path renders no doubleheader badge")

    def test_single_games_carry_no_badge(self) -> None:
        app_js = self._app_js()
        self.assertIn("game.gameInSeries = null;", app_js)

    def test_ordered_by_start_time_with_event_id_as_tiebreak(self) -> None:
        app_js = self._app_js()
        self.assertIn("if (Number.isFinite(at) && Number.isFinite(bt) && at !== bt) return at - bt;", app_js)
