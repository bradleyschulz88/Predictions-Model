"""Tests for SBR client and CLI using offline fixtures."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import mlb_sbr
import sbr_client
from sbr_client import SBRParseError, extract_next_data, get_game_rows, get_matchup

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ExtractNextDataTests(unittest.TestCase):
    def test_extracts_json_from_html_fixture(self) -> None:
        html_text = (FIXTURES / "next_data_page.html").read_text(encoding="utf-8")
        data = extract_next_data(html_text)
        self.assertEqual(data["props"]["pageProps"]["hello"], "world")

    def test_raises_when_script_missing(self) -> None:
        with self.assertRaises(SBRParseError):
            extract_next_data("<html><body>no data</body></html>")


class PagePropsHelpersTests(unittest.TestCase):
    def test_get_game_rows_from_odds_fixture(self) -> None:
        with open(FIXTURES / "odds_page.json", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = get_game_rows(data["props"]["pageProps"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["gameView"]["awayTeam"]["fullName"], "New York Yankees")

    def test_get_matchup_from_matchup_fixture(self) -> None:
        with open(FIXTURES / "matchup_page.json", encoding="utf-8") as handle:
            data = json.load(handle)
        matchup = get_matchup(data["props"]["pageProps"])
        spreads = matchup["oddsViews"]["spreadOddsViews"]
        self.assertEqual(len(spreads), 2)
        self.assertEqual(spreads[0]["sportsbook"], "DraftKings")


class CliFixtureTests(unittest.TestCase):
    def test_list_odds_json_from_fixture(self) -> None:
        fixture = str(FIXTURES / "odds_page.json")
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = mlb_sbr.main(["list-odds", "--fixture", fixture, "--format", "json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["gameCount"], 2)
        self.assertEqual(payload["games"][0]["awayTeam"], "New York Yankees")

    def test_spreads_from_fixture(self) -> None:
        fixture = str(FIXTURES / "matchup_page.json")
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = mlb_sbr.main(["spreads", "--fixture", fixture, "--format", "json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(len(payload["spreads"]), 2)

    def test_viewtypes_from_fixture(self) -> None:
        fixture = str(FIXTURES / "odds_page.json")
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = mlb_sbr.main(["viewtypes", "--fixture", fixture, "--format", "json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertIn("Spread", payload["games"][0]["viewTypes"])
        self.assertIn("MoneyLine", payload["games"][0]["viewTypes"])

    def test_inspect_odds_filters_view_types(self) -> None:
        fixture = str(FIXTURES / "odds_page.json")
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = mlb_sbr.main(
                [
                    "inspect-odds",
                    "--fixture",
                    fixture,
                    "--format",
                    "json",
                    "--limit",
                    "1",
                    "--view-filter",
                    "Spread|MoneyLine",
                ]
            )
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        view_types = {line["viewType"] for line in payload["games"][0]["lines"]}
        self.assertEqual(view_types, {"Spread", "MoneyLine"})

    def test_matchup_requires_id_or_url(self) -> None:
        buffer = io.StringIO()
        with patch("sys.stderr", buffer):
            code = mlb_sbr.main(["matchup"])
        self.assertEqual(code, 1)
        self.assertIn("Provide --id or --url", buffer.getvalue())


class UrlBuilderTests(unittest.TestCase):
    def test_build_odds_url_with_date(self) -> None:
        self.assertEqual(
            sbr_client.build_odds_url("2026-06-15"),
            "https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date=2026-06-15",
        )

    def test_build_matchup_url(self) -> None:
        self.assertEqual(
            sbr_client.build_matchup_url(368835),
            "https://www.sportsbookreview.com/scores/mlb-baseball/matchup/368835",
        )


if __name__ == "__main__":
    unittest.main()
