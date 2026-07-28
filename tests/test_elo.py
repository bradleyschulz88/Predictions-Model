"""Tests for the Elo rating engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from elo import (  # noqa: E402
    START_RATING,
    EloTable,
    build_history,
    expected_score,
    margin_multiplier,
    parse_matchup,
    rating_edge,
    results_from_accuracy,
)


class ExpectedScoreTests(unittest.TestCase):
    def test_equal_ratings_are_a_coin_flip(self) -> None:
        self.assertAlmostEqual(expected_score(1500, 1500), 0.5)

    def test_four_hundred_points_is_ten_to_one(self) -> None:
        self.assertAlmostEqual(expected_score(1900, 1500), 10 / 11, places=6)

    def test_stronger_side_is_favoured(self) -> None:
        self.assertGreater(expected_score(1600, 1500), 0.5)


class MarginTests(unittest.TestCase):
    def test_bigger_margins_move_ratings_more(self) -> None:
        self.assertGreater(margin_multiplier(10, 0), margin_multiplier(1, 0))

    def test_growth_is_dampened_not_linear(self) -> None:
        """A 20-run win must not count as twenty separate wins."""
        self.assertLess(margin_multiplier(20, 0), 20 * margin_multiplier(1, 0))

    def test_expected_blowouts_count_for_less(self) -> None:
        # Same margin, but one is a strong side beating a weak one.
        self.assertLess(margin_multiplier(10, 400), margin_multiplier(10, 0))


class TableTests(unittest.TestCase):
    def test_unseen_team_starts_at_the_base_rating(self) -> None:
        self.assertEqual(EloTable("mlb").rating("Nobody"), START_RATING)

    def test_a_seeded_team_starts_from_its_record(self) -> None:
        table = EloTable("mlb", {"Good": 0.6, "Bad": 0.4})
        self.assertGreater(table.rating("Good"), START_RATING)
        self.assertLess(table.rating("Bad"), START_RATING)

    def test_winning_raises_and_losing_lowers(self) -> None:
        table = EloTable("nba")
        table.observe("Home", "Away", 110, 100)
        self.assertGreater(table.rating("Home"), START_RATING)
        self.assertLess(table.rating("Away"), START_RATING)

    def test_ratings_are_zero_sum(self) -> None:
        table = EloTable("nba")
        table.observe("Home", "Away", 110, 100)
        self.assertAlmostEqual(
            table.rating("Home") + table.rating("Away"), 2 * START_RATING, places=6
        )

    def test_beating_a_stronger_side_is_worth_more(self) -> None:
        weak = EloTable("nba")
        weak.ratings["Opponent"] = 1300.0
        weak.observe("Us", "Opponent", 110, 100)
        gain_vs_weak = weak.rating("Us") - START_RATING

        strong = EloTable("nba")
        strong.ratings["Opponent"] = 1700.0
        strong.observe("Us", "Opponent", 110, 100)
        gain_vs_strong = strong.rating("Us") - START_RATING

        self.assertGreater(gain_vs_strong, gain_vs_weak)

    def test_baseball_moves_slower_than_basketball(self) -> None:
        """One MLB game is mostly noise; one NBA game is not."""
        mlb, nba = EloTable("mlb"), EloTable("nba")
        mlb.observe("H", "A", 5, 3)
        nba.observe("H", "A", 105, 103)
        self.assertLess(mlb.rating("H") - START_RATING, nba.rating("H") - START_RATING)

    def test_home_advantage_is_applied(self) -> None:
        table = EloTable("nba")
        self.assertGreater(table.pregame_edge("H", "A"), 0)


class MatchupTests(unittest.TestCase):
    def test_parses_away_at_home(self) -> None:
        self.assertEqual(parse_matchup("Boston Red Sox @ New York Yankees"),
                         ("New York Yankees", "Boston Red Sox"))

    def test_rejects_unparseable(self) -> None:
        for value in (None, "", "Yankees vs Red Sox", " @ "):
            self.assertIsNone(parse_matchup(value))


class HistoryTests(unittest.TestCase):
    def _accuracy(self):
        return {
            "picksByEventId": {
                "1": {"status": "graded", "league": "nba", "scheduleDate": "2026-07-01",
                      "matchup": "A @ B", "homeScore": "110", "awayScore": "100"},
                "2": {"status": "graded", "league": "nba", "scheduleDate": "2026-07-02",
                      "matchup": "B @ A", "homeScore": "99", "awayScore": "101"},
                "3": {"status": "pending", "league": "nba", "matchup": "A @ B"},
            }
        }

    def test_only_graded_results_are_used(self) -> None:
        self.assertEqual(len(results_from_accuracy(self._accuracy())), 2)

    def test_results_come_back_in_date_order(self) -> None:
        rows = results_from_accuracy(self._accuracy())
        self.assertEqual([row["date"] for row in rows], ["2026-07-01", "2026-07-02"])

    def test_pregame_edge_excludes_the_games_own_result(self) -> None:
        """Using a rating that already contains the result would leak the answer."""
        rows = results_from_accuracy(self._accuracy())
        _, pregame = build_history(rows)
        # The first game is seen with both sides still unrated, so the only
        # edge is home advantage.
        table = EloTable("nba")
        self.assertAlmostEqual(pregame["1"], table.home_advantage, places=6)

    def test_every_result_gets_a_pregame_edge(self) -> None:
        rows = results_from_accuracy(self._accuracy())
        _, pregame = build_history(rows)
        self.assertEqual(set(pregame), {"1", "2"})


class RatingEdgeTests(unittest.TestCase):
    def _ratings(self):
        return {"leagues": {"nba": {"homeAdvantage": 60.0, "ratings": {"A": 1600.0, "B": 1500.0}}}}

    def test_edge_includes_home_advantage(self) -> None:
        self.assertAlmostEqual(rating_edge(self._ratings(), "nba", "A", "B"), 160.0)

    def test_unknown_team_yields_nothing(self) -> None:
        self.assertIsNone(rating_edge(self._ratings(), "nba", "A", "Nobody"))

    def test_unknown_league_yields_nothing(self) -> None:
        self.assertIsNone(rating_edge(self._ratings(), "mlb", "A", "B"))

    def test_missing_table_yields_nothing(self) -> None:
        self.assertIsNone(rating_edge({}, "nba", "A", "B"))


if __name__ == "__main__":
    unittest.main()


class EloIsAvailableBeforePredictions(unittest.TestCase):
    """elo_ratings.json is gitignored, so a fresh CI checkout has no ratings.

    predict_game logs the pre-game rating gap as a feature, so if the table is
    only built *after* predictions run, rating_edge returns None on every game
    and eloEdge is logged as null forever -- which is exactly what happened:
    841 logged predictions, zero with an elo edge.
    """

    def test_build_runs_before_any_league_is_predicted(self) -> None:
        import inspect

        from scripts import build_pages_data

        source = inspect.getsource(build_pages_data.main)
        first_build = source.find("build_elo_ratings(")
        first_predict = source.find("build_league_payload_resilient(")
        self.assertGreater(first_build, -1, "ratings are never built")
        self.assertGreater(first_predict, -1, "nothing is predicted")
        self.assertLess(
            first_build,
            first_predict,
            "Elo ratings must be built before predictions, or eloEdge logs as null",
        )

    def test_edge_is_none_without_a_ratings_table(self) -> None:
        """The failure mode this ordering avoids, pinned so it stays understood."""
        self.assertIsNone(rating_edge({}, "mlb", "Yankees", "Tigers"))
