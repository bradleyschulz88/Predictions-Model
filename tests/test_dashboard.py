"""Tests for ESPN schedule integration and dashboard API."""

from __future__ import annotations

import json
import threading
import unittest
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
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ESPN_FIXTURE = FIXTURES / "espn_scoreboard_20260616.json"


class ESPNDataTests(unittest.TestCase):
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

    def test_default_game_date_is_tomorrow(self) -> None:
        self.assertRegex(default_game_date(), r"^\d{4}-\d{2}-\d{2}$")


WORLDCUP_FIXTURE = FIXTURES / "espn_worldcup_scoreboard_20260615.json"
AFL_FIXTURE = FIXTURES / "espn_afl_scoreboard_20260614.json"


class MultiSportTests(unittest.TestCase):
    def test_parse_worldcup_scoreboard(self) -> None:
        if not WORLDCUP_FIXTURE.is_file():
            self.skipTest("World Cup fixture not downloaded")
        with open(WORLDCUP_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="worldcup")
        self.assertGreater(len(games), 0)
        self.assertEqual(games[0]["league"], "worldcup")
        self.assertNotIn("awayPitcher", games[0])

    def test_parse_afl_scoreboard(self) -> None:
        if not AFL_FIXTURE.is_file():
            self.skipTest("AFL fixture not downloaded")
        with open(AFL_FIXTURE, encoding="utf-8") as handle:
            games = parse_scoreboard(json.load(handle), league="afl")
        self.assertGreater(len(games), 0)
        self.assertEqual(games[0]["league"], "afl")

    def test_default_game_date_worldcup_is_today(self) -> None:
        self.assertRegex(default_game_date("worldcup"), r"^\d{4}-\d{2}-\d{2}$")


class DashboardDataTests(unittest.TestCase):
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


class DashboardServerTests(unittest.TestCase):
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

    def _get_json(self, path: str) -> dict:
        with urllib.request.urlopen(f"http://{self.host}:{self.port}{path}", timeout=5) as response:
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
        self.assertIn("Sports Games Dashboard", html)


if __name__ == "__main__":
    unittest.main()
