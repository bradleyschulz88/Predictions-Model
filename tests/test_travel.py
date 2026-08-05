"""Travel and timezone burden on the visiting club."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.travel import TEAM_HOME, travel_context, travel_edge  # noqa: E402


class TableTests(unittest.TestCase):
    """Names have to match the feed exactly or the lookup silently misses.

    TEAM_HOME is keyed on the club name ESPN emits, and travel_context returns
    None for anything it does not recognise -- which is the right failure mode
    but an invisible one. A typo here reads as "no travel data", not an error,
    so these check the table against the names actually seen in the feed.
    """

    LEAGUE_SIZES = {"mlb": 30, "nba": 30, "nfl": 32, "wnba": 15}

    def test_covers_every_club_in_the_four_leagues_it_claims(self) -> None:
        self.assertEqual(len(TEAM_HOME), sum(self.LEAGUE_SIZES.values()))

    def test_every_entry_is_complete(self) -> None:
        for club, entry in TEAM_HOME.items():
            for key in ("lat", "lon", "utc", "dst"):
                self.assertIn(key, entry, club)

    def test_coordinates_are_plausible_for_north_america(self) -> None:
        """A transposed sign puts a club in the wrong hemisphere and the

        distance still computes, so nothing downstream would notice."""
        for club, entry in TEAM_HOME.items():
            self.assertTrue(20.0 < entry["lat"] < 55.0, f"{club} latitude {entry['lat']}")
            self.assertTrue(-125.0 < entry["lon"] < -65.0, f"{club} longitude {entry['lon']}")
            self.assertIn(entry["utc"], (-5, -6, -7, -8), club)

    def test_arizona_clubs_do_not_observe_daylight_saving(self) -> None:
        """The state whose offset to the east changes across the season.

        Three clubs sit in it, one per league. Missing any of them puts an
        hour of phantom body-clock shift on every summer visitor.
        """
        for club in ("Arizona Diamondbacks", "Phoenix Suns", "Arizona Cardinals",
                     "Phoenix Mercury"):
            self.assertFalse(TEAM_HOME[club]["dst"], club)

    def test_clubs_sharing_a_ground_share_its_coordinates(self) -> None:
        for one, other in (("New York Giants", "New York Jets"),
                           ("Los Angeles Chargers", "Los Angeles Rams")):
            self.assertEqual(TEAM_HOME[one], TEAM_HOME[other])
            self.assertEqual(travel_context(one, other)["distanceKm"], 0)

    def test_the_all_star_sides_are_deliberately_absent(self) -> None:
        """They appear in the WNBA feed and have no home to travel from."""
        for name in ("TEAM COOP", "TEAM SPOON"):
            self.assertNotIn(name, TEAM_HOME)
            self.assertIsNone(travel_context("Seattle Storm", name))


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

    def test_computed_for_basketball_too(self) -> None:
        """This used to be gated on league == "mlb".

        The gate predated the table covering anything but baseball, and it
        withheld the feature from the sport it should matter most in: 82 games
        with back-to-backs across three time zones, against a baseball series
        that parks a club in one city for three days.
        """
        from mlb_predictions import extract_model_inputs

        game = {
            "league": "nba", "homeTeam": "Boston Celtics", "awayTeam": "Los Angeles Lakers",
            "homeRecord": "25-25", "awayRecord": "25-25",
        }
        self.assertGreater(extract_model_inputs(game)["travelDiff"], 0)

    def test_computed_for_football_and_basketball_alike(self) -> None:
        from mlb_predictions import extract_model_inputs

        for league, home, away in (("nfl", "Buffalo Bills", "Seattle Seahawks"),
                                   ("wnba", "New York Liberty", "Seattle Storm")):
            game = {
                "league": league, "homeTeam": home, "awayTeam": away,
                "homeRecord": "5-5", "awayRecord": "5-5",
            }
            self.assertIsNotNone(extract_model_inputs(game)["travelDiff"], league)

    def test_a_league_outside_the_table_still_yields_none(self) -> None:
        """The table is the limit now, not a hardcoded league check -- so a

        sport nobody has entered venues for has to stay silent rather than
        score zero, which would read as "no trip"."""
        from mlb_predictions import extract_model_inputs

        game = {
            "league": "afl", "homeTeam": "Carlton", "awayTeam": "Essendon",
            "homeRecord": "5-5", "awayRecord": "5-5",
        }
        self.assertIsNone(extract_model_inputs(game)["travelDiff"])


if __name__ == "__main__":
    unittest.main()
