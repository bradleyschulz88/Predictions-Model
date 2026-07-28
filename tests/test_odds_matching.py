"""Tests for SBR fuzzy matching and ESPN odds fallback."""

from __future__ import annotations

import unittest

from espn_enrichment import ensure_espn_odds_on_games
from mlb_data import _find_sbr_odds_match, merge_sbr_odds_into_games
from mlb_predictions import has_moneyline_lines


class OddsMatchingTests(unittest.TestCase):
    def test_fuzzy_match_athletics_team_name(self) -> None:
        odds_by_matchup = {
            "pittsburgh pirates|oakland athletics": [
                {
                    "sportsbook": "DraftKings",
                    "viewType": "MoneyLine",
                    "currentLine": {"homeOdds": -140, "awayOdds": 115},
                }
            ]
        }
        view_types = {"pittsburgh pirates|oakland athletics": ["MoneyLine"]}
        matched = _find_sbr_odds_match("Pittsburgh Pirates", "Athletics", odds_by_matchup, view_types)
        self.assertIsNotNone(matched)
        self.assertEqual(len(matched[0]), 1)

    def test_ensure_espn_odds_fallback(self) -> None:
        game = {
            "lines": [],
            "enrichment": {
                "espnOdds": [
                    {
                        "sportsbook": "ESPN BET",
                        "viewType": "MoneyLine",
                        "currentLine": {"home": -130, "away": 110},
                    }
                ]
            },
        }
        ensure_espn_odds_on_games([game])
        self.assertTrue(has_moneyline_lines(game["lines"]))
        self.assertEqual(game["oddsSource"], "espn")

    def test_merge_sbr_skips_leagues_without_slug(self) -> None:
        games = [{"awayTeam": "A", "homeTeam": "B", "lines": []}]
        merge_sbr_odds_into_games(games, league="worldcup", date_value="2026-06-15")
        self.assertEqual(games[0]["lines"], [])


if __name__ == "__main__":
    unittest.main()


class OddsFailureDoesNotDropSchedule(unittest.TestCase):
    """The schedule is the product; odds only decorate it.

    SBR serves a valid page with no `oddsTables` whenever a league has no priced
    board. `get_game_rows` raises on that, and it used to escape
    `merge_sbr_odds_into_games` and abort the entire payload -- so NFL, NBA, WNBA
    and EPL published "0 games" on days they had a full card. MLB never showed
    the bug because SBR always prices MLB.
    """

    def _games(self) -> list[dict]:
        return [{"awayTeam": "New York Liberty", "homeTeam": "Las Vegas Aces", "lines": []}]

    def test_missing_odds_table_leaves_games_intact(self) -> None:
        from unittest.mock import patch

        from sbr_client import SBRParseError

        games = self._games()
        with patch("mlb_data.get_page_props", return_value={}), patch(
            "mlb_data.get_game_rows", side_effect=SBRParseError("oddsTables missing or empty in pageProps")
        ):
            merge_sbr_odds_into_games(games, league="wnba", date_value="2026-07-25")

        self.assertEqual(len(games), 1)
        self.assertEqual(games[0]["lines"], [])

    def test_fetch_failure_leaves_games_intact(self) -> None:
        from unittest.mock import patch

        from sbr_client import SBRFetchError

        games = self._games()
        with patch("mlb_data.get_page_props", side_effect=SBRFetchError("503")):
            merge_sbr_odds_into_games(games, league="wnba", date_value="2026-07-25")

        self.assertEqual(len(games), 1)

    def test_unpriced_league_skips_sbr_entirely(self) -> None:
        """AFL and the World Cup have no odds slug, so they must not call SBR."""
        from unittest.mock import patch

        games = self._games()
        with patch("mlb_data.get_page_props", side_effect=AssertionError("should not fetch")):
            merge_sbr_odds_into_games(games, league="afl", date_value="2026-07-25")
            merge_sbr_odds_into_games(games, league="worldcup", date_value="2026-07-25")

        self.assertEqual(len(games), 1)


class ProviderFailureDoesNotDropSchedule(unittest.TestCase):
    """Once the scoreboard parses, no downstream provider may cost us the slate."""

    def test_every_provider_down_still_publishes_games(self) -> None:
        from unittest.mock import patch

        import mlb_data
        import mlb_predictions

        slate = [{"eventId": "1", "awayTeam": "A", "homeTeam": "B", "matchup": "A @ B", "lines": []}]

        # apply_predictions re-attempts enrichment through its own import, which
        # would reach the network. Stub it so this stays hermetic.
        with patch.object(
            mlb_predictions, "enrich_games_with_providers", side_effect=RuntimeError("offline")
        ), patch.object(mlb_data, "fetch_scoreboard", return_value={"events": []}), patch.object(
            mlb_data, "parse_scoreboard", return_value=[dict(game) for game in slate]
        ), patch.object(
            mlb_data, "fetch_rolling_schedule_games", side_effect=RuntimeError("ESPN 500")
        ), patch.object(
            mlb_data, "enrich_games", side_effect=TimeoutError("read timeout")
        ), patch.object(
            mlb_data, "enrich_games_with_providers", side_effect=ValueError("bad JSON")
        ), patch.object(
            mlb_data, "merge_sbr_odds_into_games", side_effect=RuntimeError("SBR down")
        ):
            payload = mlb_data.fetch_dashboard_data(
                league="nba", date="2026-07-25", source="espn", include_odds=True, include_enrichment=True
            )

        self.assertEqual(payload["gameCount"], 1)
        # The degradation is recorded rather than swallowed, so an empty-looking
        # day can still be told apart from a broken one.
        self.assertIn("SBR odds", payload["degraded"])
        self.assertIn("ESPN enrichment", payload["degraded"])

    def test_healthy_build_records_no_degradation(self) -> None:
        from unittest.mock import patch

        import mlb_data

        with patch.object(mlb_data, "fetch_scoreboard", return_value={"events": []}), patch.object(
            mlb_data, "parse_scoreboard", return_value=[]
        ), patch.object(mlb_data, "ensure_espn_odds_on_games", return_value=None):
            payload = mlb_data.fetch_dashboard_data(
                league="nba", date="2026-07-25", source="espn", include_odds=False, include_enrichment=False
            )

        self.assertNotIn("degraded", payload)
