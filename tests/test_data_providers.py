"""Tests for external data providers."""

from __future__ import annotations

import unittest

from data_providers.derived import (
    _scoring_strength,
    compute_power_rating,
    merge_team_profile,
    parse_weather_impact,
    series_win_pct,
)
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

    def test_scoring_term_is_scaled_to_each_league(self) -> None:
        """One shared 130.0 divisor served four leagues and was basketball-sized.

        NFL clubs score about 22 a game, so it mapped the whole league into
        0.115-0.231 -- a near-constant that diluted win percentage instead of
        adding to it. Each league now normalises against its own scoring band.
        """
        # A mid-table club in each league should land near the middle, not
        # pinned to one end of the range.
        for league, ppg in (("nba", 115.0), ("wnba", 82.0), ("nfl", 22.5), ("afl", 85.0)):
            score = _scoring_strength(league, ppg)
            self.assertGreater(score, 0.3, msg=f"{league} mid-table pinned low")
            self.assertLess(score, 0.7, msg=f"{league} mid-table pinned high")

    def test_scoring_band_clamps_outliers(self) -> None:
        self.assertEqual(_scoring_strength("nfl", 60.0), 1.0)
        self.assertEqual(_scoring_strength("nfl", 2.0), 0.0)

    def test_unknown_league_has_no_scoring_band(self) -> None:
        self.assertIsNone(_scoring_strength("epl", 2.0))

    def test_power_diff_is_comparable_across_leagues(self) -> None:
        """The reason this matters: the fit gives strengthDiff a single shared
        coefficient, so the same real quality gap must produce a similar
        powerDiff in every league. Under the old divisor MLB produced 0.40
        while nba/nfl/wnba produced ~0.30 for identical inputs.
        """
        gaps = {}
        for league, home_ppg, away_ppg in (
            ("nba", 119.0, 108.0), ("nfl", 27.5, 18.0),
            ("wnba", 88.0, 78.0), ("afl", 92.0, 78.0),
        ):
            home = compute_power_rating(league=league, win_pct=0.700, goals_for_per_game=home_ppg)
            away = compute_power_rating(league=league, win_pct=0.300, goals_for_per_game=away_ppg)
            gaps[league] = home - away
        mlb_gap = (
            compute_power_rating(league="mlb", win_pct=0.700)
            - compute_power_rating(league="mlb", win_pct=0.300)
        )
        for league, gap in gaps.items():
            self.assertAlmostEqual(
                gap, mlb_gap, delta=0.10,
                msg=f"{league} powerDiff {gap:.4f} is not comparable with MLB's {mlb_gap:.4f}",
            )

    def test_scoring_actually_separates_equal_records(self) -> None:
        """Guards the guard: prove the term is not constant by construction."""
        strong = compute_power_rating(league="nfl", win_pct=0.625, goals_for_per_game=28.0)
        weak = compute_power_rating(league="nfl", win_pct=0.625, goals_for_per_game=17.0)
        self.assertGreater(strong - weak, 0.15)

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
