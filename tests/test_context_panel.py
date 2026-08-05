"""The context panel has to read to someone who does not follow baseball.

It used to render a bare signed number against a jargon label -- "Head to head
-1.00", "Elo gap -35", "Park run index +15" -- which only means anything if you
already know both the sport and the home-minus-away sign convention. A reader
cannot tell from "-1.00" which team is ahead, or that the park number is not
about either team at all.

These execute the real tile logic out of dashboard/board.js rather than
restating it, because a test that reimplements the mapping it checks would
agree with itself if the sign convention were ever flipped -- and a flipped
sign here silently names the wrong team.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOARD_JS = ROOT / "dashboard" / "board.js"

# The block that builds the tiles, lifted verbatim so the test cannot drift
# from the implementation. It starts at the numeric coercion helper on purpose:
# an earlier version of this file redefined `n` in the harness, so the test ran
# against its own copy and kept passing while production returned 0 for a null
# feature. Take the helper from the source or the test is checking itself.
BLOCK = re.compile(r"^  const n = \(v\) => \{.*?^  \];$", re.S | re.M)

HARNESS = """
const teamShort = (t) => t;
const sgn = (v, d) => (v > 0 ? "+" : v < 0 ? "\\u2212" : "") + Math.abs(v).toFixed(d);
const f = {features};
const play = {{ homeTeam: "HOMESIDE", awayTeam: "AWAYSIDE", league: "{league}" }};
{block}
console.log(JSON.stringify(items));
"""


def _node() -> str | None:
    return shutil.which("node")


def _tiles(league: str = "mlb", **features) -> dict[str, dict]:
    """Run the real tile block under node and return it keyed by label."""
    source = BOARD_JS.read_text(encoding="utf-8")
    match = BLOCK.search(source)
    assert match, "could not find the context tile block in board.js"
    script = HARNESS.format(
        features=json.dumps(features), block=match.group(0), league=league
    )
    out = subprocess.run(
        [_node(), "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return {item["kk"]: item for item in json.loads(out.stdout)}


def _shown(league: str, **features) -> list[str]:
    """Labels a card for this league would actually render.

    Mirrors production's own comparison, which lowercases. An earlier version
    compared the raw string and reported "MLB" as not-baseball -- a failure in
    the test, not the code.
    """
    tiles = _tiles(league, **features)
    return [
        k for k, v in tiles.items()
        if not v.get("leagues") or league.lower() in v["leagues"]
    ]


@unittest.skipUnless(_node(), "node is required to execute the board's tile logic")
class ContextPanelTests(unittest.TestCase):
    """Each assertion is a thing the old panel could not tell a reader."""

    def test_a_negative_rating_gap_names_the_away_side(self) -> None:
        tiles = _tiles(eloEdge=-35)
        tile = tiles["Team rating gap"]
        self.assertIn("AWAYSIDE", tile["hint"])
        self.assertNotIn("HOMESIDE", tile["hint"])

    def test_a_positive_rating_gap_names_the_home_side(self) -> None:
        tile = _tiles(eloEdge=35)["Team rating gap"]
        self.assertIn("HOMESIDE", tile["hint"])

    def test_the_rating_gap_value_carries_a_unit(self) -> None:
        self.assertIn("pts", _tiles(eloEdge=-35)["Team rating gap"]["kv"])

    def test_the_rating_gap_says_home_advantage_is_excluded(self) -> None:
        """rating_edge deliberately omits it; the panel implied otherwise."""
        self.assertIn("Home advantage is not", _tiles(eloEdge=-35)["Team rating gap"]["hint"])

    def test_the_park_number_says_it_favours_neither_team(self) -> None:
        """It is the only tile that is not about a side, which nothing said."""
        tile = _tiles(parkEdge=15)["Ballpark scoring"]
        self.assertIn("neither", tile["hint"].lower())
        self.assertNotIn("AWAYSIDE", tile["hint"])

    def test_a_pitchers_park_reads_as_low_scoring(self) -> None:
        self.assertIn("Low-scoring", _tiles(parkEdge=-6)["Ballpark scoring"]["hint"])

    def test_head_to_head_is_a_share_not_a_signed_difference(self) -> None:
        """-1.00 was the exact value that prompted the rewrite."""
        tile = _tiles(h2hDiff=-1.0)["This season's meetings"]
        self.assertEqual(tile["kv"], "HOMESIDE 0%")
        self.assertNotIn("-1", str(tile["kv"]))

    def test_head_to_head_value_and_hint_describe_the_same_side(self) -> None:
        tile = _tiles(h2hDiff=-1.0)["This season's meetings"]
        self.assertTrue(tile["kv"].startswith("HOMESIDE"))
        self.assertTrue(tile["hint"].startswith("HOMESIDE"))

    def test_a_split_series_reads_as_split(self) -> None:
        tile = _tiles(h2hDiff=0)["This season's meetings"]
        self.assertEqual(tile["kv"], "HOMESIDE 50%")
        self.assertIn("split", tile["hint"])

    def test_zero_handedness_means_neither_not_missing(self) -> None:
        """0 is a real answer here; the bare 0 read as absent data."""
        tile = _tiles(handednessDiff=0)["Left-handed starter"]
        self.assertEqual(tile["kv"], "neither")
        self.assertIsNotNone(tile["hint"])

    def test_a_lefty_edge_names_the_side_that_has_it(self) -> None:
        self.assertEqual(_tiles(handednessDiff=1)["Left-handed starter"]["kv"], "HOMESIDE")
        self.assertEqual(_tiles(handednessDiff=-1)["Left-handed starter"]["kv"], "AWAYSIDE")

    def test_paired_values_say_which_number_is_which_team(self) -> None:
        for label, features in (
            ("Injuries out", {"homeInjuryLoad": 5.75, "awayInjuryLoad": 2.0}),
            ("Days off", {"homeRest": 0, "awayRest": 3}),
        ):
            tile = _tiles(**features)[label]
            self.assertIn("HOMESIDE / AWAYSIDE", tile["hint"], label)

    def test_the_injury_tile_says_which_direction_is_good(self) -> None:
        tile = _tiles(homeInjuryLoad=5.75, awayInjuryLoad=2.0)["Injuries out"]
        self.assertIn("lower is healthier", tile["hint"])

    def test_missing_data_stays_distinguishable_from_zero(self) -> None:
        tiles = _tiles(eloEdge=None, handednessDiff=0)
        self.assertIsNone(tiles["Team rating gap"]["kv"])
        self.assertEqual(tiles["Left-handed starter"]["kv"], "neither")

    def test_every_populated_tile_explains_itself(self) -> None:
        tiles = _tiles(
            eloEdge=-35, parkEdge=15, bullpenDiff=0.8, h2hDiff=-1.0, travelDiff=3.22,
            handednessDiff=0, homeInjuryLoad=5.75, awayInjuryLoad=5.75,
            homeRest=0, awayRest=0,
        )
        self.assertEqual(len(tiles), 8)
        for label, tile in tiles.items():
            self.assertIsNotNone(tile["kv"], label)
            self.assertTrue(tile.get("hint"), f"{label} has a value but no explanation")


@unittest.skipUnless(_node(), "node is required to execute the board's tile logic")
class LeagueApplicabilityTests(unittest.TestCase):
    """A tile that can never resolve for a sport must not read as missing data.

    PARK_FACTORS holds 30 MLB parks, bullpen workload comes from the MLB
    pitching pipeline, and handedness is gated on the league outright. On an
    NBA card those render "no data", which says a feed is broken when nothing
    is -- false alarms on every game once those seasons start.

    Each tile names the leagues its table covers rather than carrying a
    baseball/not-baseball boolean. The boolean was true until TEAM_HOME grew to
    cover basketball and football: travel now resolves for four leagues while
    the park factors resolve for one, and no single flag can say both. Had it
    stayed a boolean, widening the travel table would have put an empty
    "Travel burden" tile straight back onto AFL and EPL cards.
    """

    MLB_ONLY = {"Ballpark scoring", "Bullpen freshness", "Left-handed starter"}
    # Where TEAM_HOME has venues. AFL and EPL are deliberately not in it.
    TRAVEL_LEAGUES = ("mlb", "nba", "nfl", "wnba")

    def test_baseball_keeps_every_tile(self) -> None:
        shown = _shown("mlb", eloEdge=-35, parkEdge=15, bullpenDiff=0.8,
                       travelDiff=3.2, handednessDiff=0)
        self.assertEqual(len(shown), 8)
        for label in self.MLB_ONLY | {"Travel burden"}:
            self.assertIn(label, shown)

    def test_basketball_drops_the_baseball_only_tiles(self) -> None:
        shown = _shown("nba", eloEdge=-35, homeInjuryLoad=3.0, awayInjuryLoad=1.5)
        for label in self.MLB_ONLY:
            self.assertNotIn(label, shown, f"{label} cannot resolve for nba")

    def test_travel_is_shown_wherever_there_are_venues_for_it(self) -> None:
        """It used to be hidden everywhere but baseball, which was the sport it

        mattered least in -- a baseball series parks a club in one city for
        three days, where an NBA season is 82 games of back-to-backs."""
        for league in self.TRAVEL_LEAGUES:
            self.assertIn("Travel burden", _shown(league, travelDiff=3.2), league)

    def test_travel_stays_hidden_where_there_are_none(self) -> None:
        """The regression this generalisation exists to prevent."""
        for league in ("afl", "epl", "worldcup"):
            self.assertNotIn("Travel burden", _shown(league, eloEdge=-35), league)

    def test_the_cross_sport_tiles_survive(self) -> None:
        """Dropping the inapplicable ones must not take the real ones with it."""
        for league in ("nba", "nfl", "wnba", "epl", "afl"):
            shown = _shown(league, eloEdge=-35, h2hDiff=0.33,
                           homeInjuryLoad=3.0, awayInjuryLoad=1.5, homeRest=1, awayRest=0)
            self.assertIn("Team rating gap", shown, league)
            self.assertIn("Injuries out", shown, league)
            self.assertIn("Days off", shown, league)
            self.assertIn("This season's meetings", shown, league)

    def test_an_unknown_league_gets_only_the_universal_tiles(self) -> None:
        shown = _shown("", eloEdge=-35, travelDiff=3.2, parkEdge=15)
        self.assertNotIn("Travel burden", shown)
        self.assertNotIn("Ballpark scoring", shown)
        self.assertIn("Team rating gap", shown)

    def test_the_flag_is_not_case_sensitive(self) -> None:
        self.assertIn("Travel burden", _shown("MLB", travelDiff=3.2))
        self.assertIn("Ballpark scoring", _shown("MLB", parkEdge=15))


class JargonRemovedTests(unittest.TestCase):
    """Static check, so it holds even where node is unavailable."""

    def setUp(self) -> None:
        self.source = BOARD_JS.read_text(encoding="utf-8")

    def test_the_old_unexplained_labels_are_gone(self) -> None:
        for label in ('"Elo gap"', '"Park run index"', '"Head to head"', '"Handedness"'):
            self.assertNotIn(label, self.source, f"{label} was jargon with no explanation")

    def test_the_pitching_block_defines_era(self) -> None:
        """ERA appeared four times as a label and was never defined."""
        self.assertIn("runs allowed per nine innings", self.source)


if __name__ == "__main__":
    unittest.main()
