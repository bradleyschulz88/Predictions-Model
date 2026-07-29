"""Travel and timezone burden on the visiting club."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.travel import TEAM_HOME, travel_context, travel_edge  # noqa: E402


class TableTests(unittest.TestCase):
    def test_covers_every_club(self) -> None:
        self.assertEqual(len(TEAM_HOME), 30)

    def test_every_entry_is_complete(self) -> None:
        for club, entry in TEAM_HOME.items():
            for key in ("lat", "lon", "utc", "dst"):
                self.assertIn(key, entry, club)

    def test_arizona_does_not_observe_daylight_saving(self) -> None:
        """The one club whose offset to the east changes across the season."""
        self.assertFalse(TEAM_HOME["Arizona Diamondbacks"]["dst"])


class DistanceTests(unittest.TestCase):
    def test_coast_to_coast_is_thousands_of_kilometres(self) -> None:
        context = travel_context("New York Yankees", "Seattle Mariners")
        self.assertGreater(context["distanceKm"], 3500)

    def test_a_division_rival_is_a_short_trip(self) -> None:
        context = travel_context("Los Angeles Dodgers", "San Diego Padres")
        self.assertLess(context["distanceKm"], 250)

    def test_distance_is_symmetric(self) -> None:
        there = travel_context("New York Yankees", "Seattle Mariners")["distanceKm"]
        back = travel_context("Seattle Mariners", "New York Yankees")["distanceKm"]
        self.assertEqual(there, back)


class TimezoneTests(unittest.TestCase):
    def test_westward_visitors_travel_east(self) -> None:
        """Seattle visiting New York moves three zones east."""
        context = travel_context("New York Yankees", "Seattle Mariners")
        self.assertEqual(context["direction"], "east")
        self.assertAlmostEqual(context["timezoneShift"], 3.0)

    def test_the_reverse_trip_is_westward(self) -> None:
        context = travel_context("Seattle Mariners", "New York Yankees")
        self.assertEqual(context["direction"], "west")
        self.assertAlmostEqual(context["timezoneShift"], -3.0)

    def test_same_zone_is_flagged_as_such(self) -> None:
        self.assertEqual(travel_context("Boston Red Sox", "New York Yankees")["direction"], "same")

    def test_eastbound_costs_more_than_the_same_trip_west(self) -> None:
        """The asymmetry is the point: flying east shortens the day."""
        east = travel_context("New York Yankees", "Seattle Mariners")["homeEdge"]
        west = travel_context("Seattle Mariners", "New York Yankees")["homeEdge"]
        self.assertGreater(east, west)

    def test_arizona_offset_moves_with_daylight_saving(self) -> None:
        summer = travel_context("Arizona Diamondbacks", "New York Mets", daylight_saving=True)
        winter = travel_context("Arizona Diamondbacks", "New York Mets", daylight_saving=False)
        self.assertNotAlmostEqual(summer["timezoneShift"], winter["timezoneShift"])


class EdgeTests(unittest.TestCase):
    def test_unknown_club_yields_none_not_zero(self) -> None:
        """'Could not work it out' must not read as 'no travel'."""
        self.assertIsNone(travel_context("Unknown FC", "New York Mets"))
        self.assertIsNone(travel_edge(None, "New York Mets"))

    def test_edge_favours_the_home_side(self) -> None:
        self.assertGreater(travel_edge("New York Yankees", "Seattle Mariners"), 0)

    def test_a_short_trip_is_near_zero(self) -> None:
        self.assertLess(travel_edge("Los Angeles Dodgers", "San Diego Padres"), 0.2)

    def test_documents_its_own_limitation(self) -> None:
        """It uses the visitor's home zone, not where they last played."""
        self.assertIn("last played", travel_context("New York Yankees", "Seattle Mariners")["note"])


class FeatureLogTests(unittest.TestCase):
    def test_logged_as_a_candidate(self) -> None:
        from mlb_predictions import extract_model_inputs

        game = {
            "league": "mlb", "homeTeam": "New York Yankees", "awayTeam": "Seattle Mariners",
            "homeRecord": "50-50", "awayRecord": "50-50",
        }
        self.assertGreater(extract_model_inputs(game)["travelDiff"], 0)

    def test_not_computed_outside_baseball(self) -> None:
        from mlb_predictions import extract_model_inputs

        game = {
            "league": "nba", "homeTeam": "New York Yankees", "awayTeam": "Seattle Mariners",
            "homeRecord": "50-50", "awayRecord": "50-50",
        }
        self.assertIsNone(extract_model_inputs(game)["travelDiff"])


if __name__ == "__main__":
    unittest.main()
