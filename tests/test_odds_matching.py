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
