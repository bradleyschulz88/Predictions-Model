"""Bullpen workload, plate umpire and pitcher handedness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_providers.bullpen as bullpen  # noqa: E402
import data_providers.matchup_context as matchup  # noqa: E402


class PitcherHandTests(unittest.TestCase):
    """Two APIs feed this pipeline and they spell handedness differently."""

    def test_stats_api_nested_shape(self) -> None:
        self.assertEqual(matchup.pitcher_hand({"pitchHand": {"code": "L"}}), "L")

    def test_flat_string_shape(self) -> None:
        self.assertEqual(matchup.pitcher_hand({"pitchHand": "R"}), "R")

    def test_espn_style_keys(self) -> None:
        self.assertEqual(matchup.pitcher_hand({"throws": "Left"}), "L")
        self.assertEqual(matchup.pitcher_hand({"hand": "right"}), "R")

    def test_unknown_or_switch_yields_none(self) -> None:
        self.assertIsNone(matchup.pitcher_hand({"throws": "S"}))
        self.assertIsNone(matchup.pitcher_hand({}))
        self.assertIsNone(matchup.pitcher_hand(None))


class HandednessTests(unittest.TestCase):
    def _game(self, home: str | None, away: str | None) -> dict:
        return {
            "league": "mlb",
            "homePitcher": {"pitchHand": {"code": home}} if home else {},
            "awayPitcher": {"pitchHand": {"code": away}} if away else {},
        }

    def test_opposed_arms_are_flagged(self) -> None:
        self.assertTrue(matchup.handedness_matchup(self._game("L", "R"))["opposed"])
        self.assertFalse(matchup.handedness_matchup(self._game("R", "R"))["opposed"])

    def test_needs_both_starters(self) -> None:
        """One known arm is not a matchup."""
        self.assertIsNone(matchup.handedness_matchup(self._game("L", None)))
        self.assertIsNone(matchup.handedness_matchup(self._game(None, None)))

    def test_diff_isolates_the_scarcer_arm(self) -> None:
        self.assertEqual(matchup.handedness_diff(self._game("L", "R")), 1.0)
        self.assertEqual(matchup.handedness_diff(self._game("R", "L")), -1.0)
        self.assertEqual(matchup.handedness_diff(self._game("R", "R")), 0.0)
        self.assertEqual(matchup.handedness_diff(self._game("L", "L")), 0.0)

    def test_applies_no_platoon_adjustment(self) -> None:
        """Doing that properly needs lineup splits, which confirm too late."""
        self.assertIn("No platoon adjustment", matchup.handedness_matchup(self._game("L", "R"))["note"])

    def test_reaches_the_feature_log(self) -> None:
        from mlb_predictions import extract_model_inputs

        game = self._game("L", "R")
        game.update({"homeTeam": "A", "awayTeam": "B", "homeRecord": "50-50", "awayRecord": "50-50"})
        self.assertEqual(extract_model_inputs(game)["handednessDiff"], 1.0)


class UmpireTests(unittest.TestCase):
    def _payload(self, officials: list[dict]) -> dict:
        return {"liveData": {"boxscore": {"officials": officials}}}

    def test_finds_the_plate_umpire(self) -> None:
        payload = self._payload([
            {"officialType": "First Base", "official": {"fullName": "Someone Else"}},
            {"officialType": "Home Plate", "official": {"fullName": "Angel Hernandez"}},
        ])
        with patch.object(matchup, "_fetch", return_value=payload):
            self.assertEqual(matchup.fetch_plate_umpire(747123), "Angel Hernandez")

    def test_unassigned_crew_is_not_an_error(self) -> None:
        """Assignments are often absent until hours before first pitch."""
        with patch.object(matchup, "_fetch", return_value=self._payload([])):
            self.assertIsNone(matchup.fetch_plate_umpire(747123))

    def test_a_failed_fetch_never_raises(self) -> None:
        with patch.object(matchup, "_fetch", side_effect=RuntimeError("statsapi down")):
            self.assertIsNone(matchup.fetch_plate_umpire(747123))

    def test_missing_game_id_short_circuits(self) -> None:
        self.assertIsNone(matchup.fetch_plate_umpire(None))


class BullpenTests(unittest.TestCase):
    def _log(self, rows: list[dict]) -> dict:
        return {"stats": [{"splits": [{"stat": row} for row in rows]}]}

    def test_starter_innings_are_excluded(self) -> None:
        """A complete game rests a bullpen; counting team innings inverts that."""
        payload = self._log([{"inningsPitched": 9.0, "startersInningsPitched": 9.0}])
        with patch.object(bullpen, "_fetch", return_value=payload):
            self.assertEqual(bullpen.team_relief_innings(147, days=1), 0.0)

    def test_relief_innings_accumulate(self) -> None:
        payload = self._log([
            {"inningsPitched": 9.0, "startersInningsPitched": 5.0},
            {"inningsPitched": 9.0, "startersInningsPitched": 6.0},
        ])
        with patch.object(bullpen, "_fetch", return_value=payload):
            self.assertEqual(bullpen.team_relief_innings(147, days=2), 7.0)

    def test_fatigue_is_centred_on_a_normal_stretch(self) -> None:
        """Zero must mean 'ordinary', not 'no innings at all'."""
        normal = bullpen.TYPICAL_RELIEF_IP_PER_GAME
        payload = self._log([{"inningsPitched": normal, "startersInningsPitched": 0.0}])
        with patch.object(bullpen, "_fetch", return_value=payload):
            self.assertAlmostEqual(bullpen.bullpen_fatigue(147, days=1), 0.0, places=2)

    def test_a_failed_fetch_never_raises(self) -> None:
        with patch.object(bullpen, "_fetch", side_effect=RuntimeError("statsapi down")):
            self.assertIsNone(bullpen.team_relief_innings(147))
            self.assertIsNone(bullpen.bullpen_fatigue(147))

    def test_edge_needs_both_sides(self) -> None:
        """A one-sided figure would read as an edge when it is a data gap."""
        with patch.object(bullpen, "bullpen_fatigue", side_effect=[2.0, None]):
            self.assertIsNone(bullpen.bullpen_edge(147, 121))

    def test_a_tired_home_bullpen_favours_the_away_side(self) -> None:
        with patch.object(bullpen, "bullpen_fatigue", side_effect=[4.0, 0.0]):
            self.assertLess(bullpen.bullpen_edge(147, 121), 0)

    def test_missing_team_id_short_circuits(self) -> None:
        self.assertIsNone(bullpen.team_relief_innings(None))


if __name__ == "__main__":
    unittest.main()
