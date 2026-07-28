"""Regression tests for scoring-path correctness fixes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlb_predictions  # noqa: E402
from mlb_predictions import (  # noqa: E402
    _last_five_pct,
    _lineup_logit_adjustment,
    apply_predictions,
    extract_spread_line,
)
from shared_utils import win_pct_from_record  # noqa: E402


class SpreadSignTests(unittest.TestCase):
    """extract_spread_line promises the home spread; it must not return away's."""

    def _lines(self, current: dict) -> list[dict]:
        return [{"viewType": "Spread", "currentLine": current}]

    def test_reads_home_spread_directly(self) -> None:
        self.assertEqual(extract_spread_line(self._lines({"home": "-3.5", "away": "+3.5"})), -3.5)

    def test_flips_sign_when_only_away_is_present(self) -> None:
        # Away +7 means the home side is a 7-point favourite: -7, not +7.
        self.assertEqual(extract_spread_line(self._lines({"away": "+7"})), -7.0)

    def test_handles_unicode_minus(self) -> None:
        self.assertEqual(extract_spread_line(self._lines({"home": "−2.5"})), -2.5)

    def test_returns_none_without_spread_lines(self) -> None:
        self.assertIsNone(extract_spread_line([{"viewType": "MoneyLine", "currentLine": {}}]))


class LastFiveSentinelTests(unittest.TestCase):
    """A -1.0 sentinel reaching the logit is indistinguishable from real form."""

    def test_zero_zero_record_returns_none(self) -> None:
        self.assertIsNone(_last_five_pct("0-0"))

    def test_unparseable_record_returns_none(self) -> None:
        self.assertIsNone(_last_five_pct("no games"))
        self.assertIsNone(_last_five_pct(None))

    def test_normal_record_still_parses(self) -> None:
        self.assertAlmostEqual(_last_five_pct("3-2"), 0.6)

    def test_never_returns_negative(self) -> None:
        for record in ("0-0", "0-0-0", "", "—", "5-0"):
            value = _last_five_pct(record)
            if value is not None:
                self.assertGreaterEqual(value, 0.0, msg=record)


class RecordFormatTests(unittest.TestCase):
    """Three-part records mean different things in soccer and football."""

    def test_soccer_reads_middle_column_as_draws(self) -> None:
        # 10W 5D 2L -> (10 + 2.5) / 17
        self.assertAlmostEqual(win_pct_from_record("10-5-2", league="epl"), 12.5 / 17)

    def test_nfl_reads_last_column_as_ties(self) -> None:
        # 10W 5L 2T -> (10 + 1) / 17
        self.assertAlmostEqual(win_pct_from_record("10-5-2", league="nfl"), 11.0 / 17)

    def test_two_part_record_unaffected_by_league(self) -> None:
        for league in (None, "mlb", "epl", "nfl"):
            self.assertAlmostEqual(win_pct_from_record("10-5", league=league), 10 / 15)

    def test_zero_games_falls_back_to_default(self) -> None:
        self.assertEqual(win_pct_from_record("0-0-0", 0.42, league="epl"), 0.42)


class LineupScaleTests(unittest.TestCase):
    """Comparing two different metrics manufactures edge from data coverage."""

    def test_no_edge_when_sides_use_different_metrics(self) -> None:
        # Home has confirmed batting averages, away only a season OPS proxy.
        game = {
            "league": "mlb",
            "homeLineup": {"batters": [{"order": index, "avg": 0.250} for index in range(1, 10)]},
            "awayLineup": {},
        }
        enrichment = {"awayAdvanced": {"opsProxy": 0.720}}
        self.assertEqual(_lineup_logit_adjustment(game, "mlb", enrichment), 0.0)

    def test_edge_when_both_sides_use_the_same_metric(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.780},
            "awayAdvanced": {"opsProxy": 0.680},
        }
        adjustment = _lineup_logit_adjustment(game, "mlb", enrichment)
        self.assertGreater(adjustment, 0.0)
        self.assertLessEqual(adjustment, 0.35)

    def test_league_average_inputs_produce_no_edge(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"opsProxy": 0.720},
            "awayAdvanced": {"opsProxy": 0.720},
        }
        self.assertAlmostEqual(_lineup_logit_adjustment(game, "mlb", enrichment), 0.0)

    def test_missing_data_on_both_sides_is_neutral(self) -> None:
        game = {"league": "mlb", "homeLineup": {}, "awayLineup": {}}
        self.assertEqual(_lineup_logit_adjustment(game, "mlb", {}), 0.0)

    def test_adjustment_stays_clamped_for_extreme_inputs(self) -> None:
        game = {"league": "nba", "homeLineup": {}, "awayLineup": {}}
        enrichment = {
            "homeAdvanced": {"pointsPerGame": 200.0},
            "awayAdvanced": {"pointsPerGame": 40.0},
        }
        self.assertLessEqual(_lineup_logit_adjustment(game, "nba", enrichment), 0.35)


class PublishableIdentityTests(unittest.TestCase):
    """Ranking must track object identity, not dict equality."""

    def test_identical_games_are_ranked_independently(self) -> None:
        def make_game() -> dict:
            return {
                "league": "mlb",
                "eventId": "same",
                "homeTeam": "Team A",
                "awayTeam": "Team B",
                "homeRecord": "60-40",
                "awayRecord": "40-60",
                "enrichment": {},
                "lines": [],
            }

        # Enrichment reaches the network; this test is about ranking only.
        with mock.patch.object(
            mlb_predictions, "enrich_games_with_providers", side_effect=lambda games, **_: games
        ):
            games = apply_predictions([make_game(), make_game()])

        # Two games that serialise identically must both keep their rank rather
        # than the second being demoted by an equality-based membership test.
        ranks = [game.get("predictionRank") for game in games]
        publishable = [(game.get("prediction") or {}).get("publishable") for game in games]
        self.assertEqual(len([rank for rank in ranks if rank is not None]), sum(1 for flag in publishable if flag))


if __name__ == "__main__":
    unittest.main()
