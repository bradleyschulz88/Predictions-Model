"""Tests for MLB park factors."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.park_factors import (  # noqa: E402
    NEUTRAL_PARK_FACTOR,
    PARK_FACTORS,
    is_neutral_site,
    park_factor,
    park_run_environment,
)


class ParkTableTests(unittest.TestCase):
    def test_covers_every_club(self) -> None:
        """A missing club silently becomes 'unknown' on real games."""
        self.assertEqual(len(PARK_FACTORS), 30)

    def test_the_extremes_are_the_right_way_round(self) -> None:
        """Coors is the largest park effect in the sport; Seattle suppresses."""
        self.assertEqual(max(PARK_FACTORS, key=PARK_FACTORS.get), "Colorado Rockies")
        self.assertGreater(PARK_FACTORS["Colorado Rockies"], 110)
        self.assertLess(PARK_FACTORS["Seattle Mariners"], NEUTRAL_PARK_FACTOR)
        self.assertLess(PARK_FACTORS["San Francisco Giants"], NEUTRAL_PARK_FACTOR)

    def test_values_stay_in_a_plausible_range(self) -> None:
        """A typo here would quietly distort every total at that park."""
        for club, value in PARK_FACTORS.items():
            self.assertGreaterEqual(value, 85.0, club)
            self.assertLessEqual(value, 125.0, club)


class ParkLookupTests(unittest.TestCase):
    def test_unknown_club_is_none_not_neutral(self) -> None:
        """'No idea' must stay distinguishable from 'we know it is average'."""
        self.assertIsNone(park_factor("Sheffield Wednesday"))
        self.assertIsNone(park_factor(None))

    def test_neutral_site_overrides_the_home_club(self) -> None:
        """A London Series game is not played in the home club's park."""
        self.assertIsNotNone(park_factor("New York Mets"))
        self.assertIsNone(park_factor("New York Mets", "London Stadium"))
        self.assertIsNone(park_factor("Chicago Cubs", "Tokyo Dome"))

    def test_normal_venue_is_not_treated_as_neutral(self) -> None:
        self.assertFalse(is_neutral_site("Coors Field"))
        self.assertFalse(is_neutral_site(None))
        self.assertEqual(park_factor("Colorado Rockies", "Coors Field"), 115.0)


class RunEnvironmentTests(unittest.TestCase):
    def test_edge_is_centred_on_zero(self) -> None:
        """A neutral park must contribute nothing, not a constant."""
        neutral = park_run_environment("Kansas City Royals")
        self.assertEqual(neutral["edge"], 0.0)
        self.assertEqual(neutral["multiplier"], 1.0)

    def test_hitters_and_pitchers_parks_get_opposite_signs(self) -> None:
        self.assertGreater(park_run_environment("Colorado Rockies")["edge"], 0)
        self.assertLess(park_run_environment("Seattle Mariners")["edge"], 0)

    def test_notes_describe_the_park(self) -> None:
        self.assertIn("hitters", park_run_environment("Colorado Rockies")["note"])
        self.assertIn("pitchers", park_run_environment("Seattle Mariners")["note"])
        self.assertIn("neutral", park_run_environment("Kansas City Royals")["note"])

    def test_unknown_park_yields_nothing(self) -> None:
        self.assertIsNone(park_run_environment("Sheffield Wednesday"))


class TotalsIntegrationTests(unittest.TestCase):
    """Park factor has to reach the totals lean, and only for baseball."""

    def _game(self, home: str, league: str = "mlb") -> dict:
        return {
            "league": league,
            "homeTeam": home,
            "awayTeam": "Visitors",
            "venueName": "Home Park",
        }

    def _lines(self) -> list[dict]:
        return [{"viewType": "Total", "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-110)"}}]

    def _over_pct(self, home: str, league: str = "mlb") -> float:
        from mlb_predictions import predict_total

        result = predict_total(self._game(home, league), self._lines(), {})
        return result["overPct"] if "overPct" in result else result["probabilities"]["over"]

    def test_a_hitters_park_leans_more_over_than_a_pitchers_park(self) -> None:
        self.assertGreater(self._over_pct("Colorado Rockies"), self._over_pct("Seattle Mariners"))

    def test_a_neutral_park_sits_between_them(self) -> None:
        neutral = self._over_pct("Kansas City Royals")
        self.assertLess(self._over_pct("Seattle Mariners"), neutral)
        self.assertLess(neutral, self._over_pct("Colorado Rockies"))

    def test_park_factor_is_not_applied_outside_baseball(self) -> None:
        """The table is MLB clubs; another league must not match by accident."""
        from mlb_predictions import predict_total

        result = predict_total(self._game("Colorado Rockies", "nba"), self._lines(), {})
        details = " ".join(result.get("reasons") or []) + str(result.get("detail") or "")
        self.assertNotIn("runs index", details)


class LoggedFeatureTests(unittest.TestCase):
    """parkEdge is logged so it can be judged later, and backfilled from the log."""

    def test_park_edge_is_extracted_as_a_feature(self) -> None:
        from mlb_predictions import extract_prediction_features

        game = {
            "league": "mlb",
            "homeTeam": "Colorado Rockies",
            "awayTeam": "San Diego Padres",
            "homeRecord": "50-50",
            "awayRecord": "50-50",
        }
        features = extract_prediction_features(game, {})
        self.assertEqual(features["parkEdge"], 15.0)

    def test_unknown_park_logs_none_rather_than_zero(self) -> None:
        from mlb_predictions import extract_prediction_features

        game = {"league": "mlb", "homeTeam": "Unknown FC", "awayTeam": "B",
                "homeRecord": "50-50", "awayRecord": "50-50"}
        self.assertIsNone(extract_prediction_features(game, {})["parkEdge"])


if __name__ == "__main__":
    unittest.main()
