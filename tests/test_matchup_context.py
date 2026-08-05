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
    """Relief innings come from the boxscore, the only endpoint that has the split.

    The team game log returns 59 pitching fields and none of them separates
    starters from relievers -- a production build printed the whole list. The
    first version of this feature tried to subtract starters out of that endpoint
    and reported nothing on every game.
    """

    def _box(self, pitchers: list[int], innings: dict[int, float], team_id: int = 147) -> dict:
        players = {
            f"ID{pid}": {"stats": {"pitching": {"inningsPitched": innings.get(pid, 0.0)}}}
            for pid in pitchers
        }
        return {
            "teams": {
                "home": {"team": {"id": team_id}, "pitchers": pitchers, "players": players},
                "away": {"team": {"id": 999}, "pitchers": [], "players": {}},
            }
        }

    def test_the_starter_is_excluded(self) -> None:
        """Index 0 is the starter; everyone after it is relief."""
        box = self._box([1, 2, 3], {1: 6.0, 2: 2.0, 3: 1.0})
        with patch.object(bullpen, "_fetch", return_value=box):
            self.assertEqual(bullpen.relief_innings_in_game("777", 147), 3.0)

    def test_a_complete_game_is_zero_not_unknown(self) -> None:
        """The bullpen rested. Zero is data; None would discard the row."""
        box = self._box([1], {1: 9.0})
        with patch.object(bullpen, "_fetch", return_value=box):
            self.assertEqual(bullpen.relief_innings_in_game("777", 147), 0.0)

    def test_the_other_club_in_the_game_is_not_counted(self) -> None:
        box = self._box([1, 2], {1: 5.0, 2: 4.0}, team_id=147)
        with patch.object(bullpen, "_fetch", return_value=box):
            self.assertIsNone(bullpen.relief_innings_in_game("777", 555))

    def test_an_empty_pitcher_list_is_unknown(self) -> None:
        with patch.object(bullpen, "_fetch", return_value=self._box([], {})):
            self.assertIsNone(bullpen.relief_innings_in_game("777", 147))

    def test_workload_sums_across_recent_games(self) -> None:
        box = self._box([1, 2], {1: 6.0, 2: 3.0})
        with patch.object(bullpen, "_recent_game_pks", return_value=["1", "2", "3"]), \
             patch.object(bullpen, "_fetch", return_value=box):
            self.assertEqual(bullpen.team_relief_innings(147, days=3), 9.0)

    def test_a_short_window_is_scaled_not_treated_as_rest(self) -> None:
        """Two games in three days must not read as a rested bullpen."""
        box = self._box([1, 2], {1: 6.0, 2: 3.0})
        with patch.object(bullpen, "_recent_game_pks", return_value=["1", "2"]), \
             patch.object(bullpen, "_fetch", return_value=box):
            self.assertEqual(bullpen.team_relief_innings(147, days=3), 9.0)

    def test_fatigue_is_centred_on_a_normal_stretch(self) -> None:
        """Zero must mean 'ordinary', not 'no innings at all'."""
        normal = bullpen.TYPICAL_RELIEF_IP_PER_GAME
        box = self._box([1, 2], {1: 5.0, 2: normal})
        with patch.object(bullpen, "_recent_game_pks", return_value=["1"]), \
             patch.object(bullpen, "_fetch", return_value=box):
            self.assertAlmostEqual(bullpen.bullpen_fatigue(147, days=1), 0.0, places=2)

    def test_workload_actually_varies(self) -> None:
        """Guards the guard: prove the metric is not constant by construction.

        The previous version returned exactly the typical figure every time, so
        an 18-inning day and a 9-inning day both scored 0.00 fatigue.
        """
        light = self._box([1, 2], {1: 8.0, 2: 1.0})
        heavy = self._box([1, 2, 3, 4], {1: 1.0, 2: 3.0, 3: 3.0, 4: 2.0})
        with patch.object(bullpen, "_recent_game_pks", return_value=["1"]):
            with patch.object(bullpen, "_fetch", return_value=light):
                easy = bullpen.bullpen_fatigue(147, days=1)
            with patch.object(bullpen, "_fetch", return_value=heavy):
                hard = bullpen.bullpen_fatigue(147, days=1)
        self.assertLess(easy, hard)

    def test_a_failed_fetch_never_raises(self) -> None:
        with patch.object(bullpen, "_fetch", side_effect=RuntimeError("statsapi down")):
            self.assertIsNone(bullpen.team_relief_innings(147))
            self.assertIsNone(bullpen.bullpen_fatigue(147))

    def test_no_completed_games_is_not_an_error(self) -> None:
        with patch.object(bullpen, "_recent_game_pks", return_value=[]):
            self.assertIsNone(bullpen.team_relief_innings(147))

    def test_only_final_games_are_used(self) -> None:
        schedule = {"dates": [
            {"date": "2026-07-28", "games": [
                {"gamePk": 1, "status": {"abstractGameState": "Final"}},
                {"gamePk": 2, "status": {"abstractGameState": "Live"}},
                {"gamePk": 3, "status": {"abstractGameState": "Preview"}},
            ]},
        ]}
        with patch.object(bullpen, "_fetch", return_value=schedule):
            self.assertEqual(bullpen._recent_game_pks(147, days=3), ["1"])

    def test_edge_needs_both_sides(self) -> None:
        """A one-sided figure would read as an edge when it is a data gap."""
        with patch.object(bullpen, "bullpen_fatigue", side_effect=[2.0, None]):
            self.assertIsNone(bullpen.bullpen_edge(147, 121))

    def test_a_tired_home_bullpen_favours_the_away_side(self) -> None:
        with patch.object(bullpen, "bullpen_fatigue", side_effect=[4.0, 0.0]):
            self.assertLess(bullpen.bullpen_edge(147, 121), 0)

    def test_missing_team_id_short_circuits(self) -> None:
        self.assertIsNone(bullpen.team_relief_innings(None))


class DeadFeatureRegressionTests(unittest.TestCase):
    """Three candidates that logged a value on every game and meant nothing.

    A real build produced 120 rows: h2hDiff and handednessDiff were None on all
    of them, and bullpenDiff was exactly 0.0 on all 99 it filled. A feature that
    is constant or always-absent cannot be detected as broken by the ablation --
    it simply never ships, looking like a fair test that it lost.
    """

    def test_handedness_has_a_data_source(self) -> None:
        """ESPN carries name, id, ERA and record -- and no arm."""
        from data_providers.mlb_pitcher import _pitcher_hand

        payload = {"people": [{"pitchHand": {"code": "L"}}]}
        with patch("data_providers.mlb_pitcher._fetch_api", return_value=payload):
            self.assertEqual(_pitcher_hand(669373), "L")

    def test_handedness_lookup_never_raises(self) -> None:
        from data_providers.mlb_pitcher import _pitcher_hand

        with patch("data_providers.mlb_pitcher._fetch_api", side_effect=RuntimeError("down")):
            self.assertIsNone(_pitcher_hand(669373))

    def test_handedness_reaches_the_pitcher_the_feature_reads(self) -> None:
        """The lookup is useless unless it lands where pitcher_hand looks."""
        from data_providers.mlb_pitcher import enrich_mlb_pitching_context

        game = {
            "league": "mlb", "homeTeam": "Dodgers", "awayTeam": "Padres",
            "homePitcher": {"name": "A"}, "awayPitcher": {"name": "B"},
        }
        with patch("data_providers.mlb_pitcher._resolve_pitcher_id", return_value=1), \
             patch("data_providers.mlb_pitcher._pitcher_season_stats", return_value={}), \
             patch("data_providers.mlb_pitcher._pitcher_recent_start_era", return_value=None), \
             patch("data_providers.mlb_pitcher._resolve_team_id", return_value=None), \
             patch("data_providers.mlb_pitcher._pitcher_hand", side_effect=["L", "R"]):
            enrich_mlb_pitching_context(game)

        self.assertEqual(matchup.pitcher_hand(game["homePitcher"]), "L")
        self.assertEqual(matchup.pitcher_hand(game["awayPitcher"]), "R")
        self.assertEqual(matchup.handedness_diff(game), 1.0)


class PitcherFipTests(unittest.TestCase):
    """FIP was the fourth always-None candidate, and the quietest: a real
    build logged 0 FIP values across 872 graded games while ERA filled in on
    97-100% of them, using the same _pitcher_season_stats call. The "season"
    stat type's pitching object never carries fip -- only "sabermetrics"
    does -- so the single request this used to make could not have returned
    it no matter how many games ran."""

    def test_fip_is_read_from_the_sabermetrics_stat_type_not_season(self) -> None:
        """The season payload here deliberately carries no fip field at all,
        matching what the real API returns for stats=season -- if the
        implementation still read fip off that response the way the old
        single-request version tried to, this would come back None."""
        from data_providers.mlb_pitcher import _pitcher_season_stats

        season = {"stats": [{"splits": [{"stat": {"era": "3.10"}}]}]}
        sabermetrics = {"stats": [{"splits": [{"stat": {"fip": "3.45"}}]}]}
        with patch("data_providers.mlb_pitcher._fetch_api", side_effect=[season, sabermetrics]):
            result = _pitcher_season_stats(543037)
        self.assertEqual(result, {"era": 3.10, "fip": 3.45})

    def test_a_failed_sabermetrics_call_still_returns_era(self) -> None:
        """The two stat types are fetched separately so a sabermetrics outage
        costs only fip, not the era value the first call already had."""
        from data_providers.mlb_pitcher import _pitcher_season_stats

        season = {"stats": [{"splits": [{"stat": {"era": "3.10"}}]}]}
        with patch(
            "data_providers.mlb_pitcher._fetch_api",
            side_effect=[season, RuntimeError("sabermetrics down")],
        ):
            result = _pitcher_season_stats(543037)
        self.assertEqual(result, {"era": 3.10, "fip": None})

    def test_fip_reaches_the_pitching_context_the_feature_reads(self) -> None:
        """The lookup is useless unless it lands where pitchingFipDiff looks."""
        from data_providers.mlb_pitcher import enrich_mlb_pitching_context

        game = {
            "league": "mlb", "homeTeam": "Dodgers", "awayTeam": "Padres",
            "homePitcher": {"name": "A"}, "awayPitcher": {"name": "B"},
        }
        with patch("data_providers.mlb_pitcher._resolve_pitcher_id", side_effect=[1, 2]), \
             patch("data_providers.mlb_pitcher._pitcher_season_stats",
                   side_effect=[{"era": 3.0, "fip": 3.2}, {"era": 4.0, "fip": 4.4}]), \
             patch("data_providers.mlb_pitcher._pitcher_recent_start_era", return_value=None), \
             patch("data_providers.mlb_pitcher._resolve_team_id", return_value=None), \
             patch("data_providers.mlb_pitcher._pitcher_hand", return_value=None):
            context = enrich_mlb_pitching_context(game)

        self.assertEqual(context["homePitcherFip"], 3.2)
        self.assertEqual(context["awayPitcherFip"], 4.4)


class HeadToHeadResolutionTests(unittest.TestCase):
    """h2hDiff was None on all 120 rows of a real build.

    `series_win_pct` only resolves for a club named in ESPN's summary string,
    and the commonest summary -- "Series tied 1-1" -- names neither, while
    "Dodgers lead series 2-1" names only one. Requiring both to resolve
    independently meant the pair almost never did.
    """

    def _diff(self, **h2h):
        from mlb_predictions import _h2h_diff

        return _h2h_diff({"headToHead": h2h})

    def test_one_named_club_determines_the_other(self) -> None:
        self.assertAlmostEqual(self._diff(homeSeriesWinPct=0.75, awaySeriesWinPct=None), 0.5)
        self.assertAlmostEqual(self._diff(homeSeriesWinPct=None, awaySeriesWinPct=0.75), -0.5)

    def test_a_tied_series_names_neither_club(self) -> None:
        self.assertEqual(
            self._diff(homeSeriesWinPct=None, awaySeriesWinPct=None, seriesScore="1-1"), 0.0
        )

    def test_both_sides_still_work(self) -> None:
        self.assertAlmostEqual(self._diff(homeSeriesWinPct=0.6, awaySeriesWinPct=0.4), 0.2)

    def test_an_uneven_unnamed_series_is_not_guessed(self) -> None:
        """Nothing says which club holds the 2, so it stays unknown."""
        self.assertIsNone(
            self._diff(homeSeriesWinPct=None, awaySeriesWinPct=None, seriesScore="2-1")
        )

    def test_no_series_is_still_none(self) -> None:
        self.assertIsNone(self._diff())


class SelfDiagnosingTests(unittest.TestCase):
    """A dead feature has to announce itself, not fail quietly.

    The previous warning printed the field names the game log returned, which is
    what proved that endpoint could never work: 59 pitching fields, none of them
    a starter/reliever split. It has done its job and is replaced by one about
    the boxscore, which is where the split actually lives.
    """

    def test_an_unreadable_boxscore_warns_once(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        empty = {"teams": {"home": {"team": {"id": 147}, "pitchers": [], "players": {}}}}
        with patch.object(bullpen, "_recent_game_pks", return_value=["1", "2"]), \
             patch.object(bullpen, "_fetch", return_value=empty), redirect_stdout(buffer):
            bullpen._warned = False
            for _ in range(4):
                bullpen.team_relief_innings(147)

        output = buffer.getvalue()
        self.assertIn("Bullpen workload", output)
        self.assertEqual(output.count("Bullpen workload"), 1, "once per run, not per game")

    def test_the_warning_explains_why_the_game_log_cannot_substitute(self) -> None:
        """So nobody retries the endpoint that was already proven unusable."""
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        empty = {"teams": {"home": {"team": {"id": 147}, "pitchers": [], "players": {}}}}
        with patch.object(bullpen, "_recent_game_pks", return_value=["1"]), \
             patch.object(bullpen, "_fetch", return_value=empty), redirect_stdout(buffer):
            bullpen._warned = False
            bullpen.team_relief_innings(147)
        self.assertIn("game log cannot substitute", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
