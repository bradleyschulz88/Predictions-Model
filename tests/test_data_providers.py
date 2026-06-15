"""Tests for external data providers."""

from __future__ import annotations

import unittest

from data_providers.derived import compute_power_rating, merge_team_profile, parse_weather_impact, series_win_pct
from data_providers.utils import best_team_match, normalize_team_name
from mlb_predictions import predict_game


class ProviderUtilsTests(unittest.TestCase):
    def test_normalize_and_match_team(self) -> None:
        self.assertEqual(normalize_team_name("New York Yankees"), "new york yankees")
        candidates = {
            "new york yankees": {"id": "1"},
            "boston red sox": {"id": "2"},
        }
        self.assertEqual(best_team_match("Yankees", candidates), "new york yankees")


class DerivedMetricsTests(unittest.TestCase):
    def test_weather_impact_warm(self) -> None:
        impact = parse_weather_impact("82°F, 5% precipitation, 8 mph wind")
        self.assertIsNotNone(impact)
        assert impact is not None
        self.assertGreater(impact["runEnvironmentAdj"], 0)

    def test_series_win_pct(self) -> None:
        pct = series_win_pct({"summary": "Yankees lead series", "seriesScore": "4-2"}, "New York Yankees")
        self.assertIsNotNone(pct)

    def test_power_rating_mlb(self) -> None:
        rating = compute_power_rating(league="mlb", win_pct=0.6, run_diff_per_game=0.8, form_pct=0.7)
        self.assertIsNotNone(rating)
        assert rating is not None
        self.assertGreater(rating, 0.5)

    def test_merge_team_profile(self) -> None:
        profile = merge_team_profile(
            league="mlb",
            espn_stats={"onBasePct": 0.33, "sluggingPct": 0.43, "era": 3.8},
            espn_standings={"winPct": 0.58, "pointsPerGame": None, "goalsAgainstPerGame": None},
            mlb_official={"runDifferential": 40, "gamesPlayed": 100, "winPct": 0.58},
            form_pct=0.6,
        )
        self.assertIn("powerRating", profile)
        self.assertIn("MLB.com", profile["sources"])


class PredictionAdvancedTests(unittest.TestCase):
    def test_predict_game_uses_advanced_profile(self) -> None:
        game = {
            "league": "mlb",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeRecord": "50-30",
            "awayRecord": "35-45",
            "lines": [],
            "enrichment": {
                "homeAdvanced": {"powerRating": 0.72, "runDifferential": 50},
                "awayAdvanced": {"powerRating": 0.48, "runDifferential": -20},
                "restDays": {"home": 2, "away": 0},
                "headToHead": {"homeSeriesWinPct": 0.67, "awaySeriesWinPct": 0.33, "summary": "Home leads 2-1"},
                "sources": ["ESPN", "MLB.com"],
            },
        }
        prediction = predict_game(game)
        labels = [factor["label"] for factor in prediction["factors"]]
        self.assertIn("Advanced team profile", labels)
        self.assertIn("Rest days", labels)
        self.assertIn("Season series", labels)
        self.assertGreater(prediction["confidence"], 50)


if __name__ == "__main__":
    unittest.main()
