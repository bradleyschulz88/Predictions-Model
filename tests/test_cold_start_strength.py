"""A record earns its weight from its sample size.

Found 13 Aug 2026, the week NFL preseason started. The model was publishing
92.3% confidence on Colts @ Patriots where the market implied 41.2%, 95.0% on
Panthers @ Cardinals against 45.5%, and 87.1% on Packers @ Steelers against
45.5% -- one of them pinned to the MAX_PROB clamp, on a competition with a
single graded result behind it.

Divergence by league, current pipeline:

    NFL    n= 10   median 27.2pts   70% over 15
    MLB    n=658   median  8.2pts   25% over 15
    WNBA   n= 50   median  4.5pts    6% over 15
    AFL    n=  6   median  5.6pts    0% over 15

The original review of this project opened on median 25 points with 77% over
15. That was fixed, in baseball. NFL preseason was running the pre-fix model
and nothing noticed, because the published divergence figure is pooled and
baseball dominates it.

The mechanism was visible in the features. `recordDiff` is None in August, so
strength fell through to `powerRating` computed from the one or two preseason
games already played: homePower 0.80 against awayPower 0.00 is one club that
won its opener against one that lost. `MIN_GAMES_BEFORE_USABLE = 10` guards the
fit; nothing guarded the feature.

Three changes, all general rather than NFL-specific, because the same defect
arrives in MLB every April and the NBA every October.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_fit import STRENGTH_SHRINK_GAMES, build_feature_dict  # noqa: E402
from shared_utils import games_played  # noqa: E402


class GamesPlayedTests(unittest.TestCase):
    """The sample size the model could not previously ask for."""

    def test_it_counts_the_games_in_a_record(self) -> None:
        self.assertEqual(games_played("1-0"), 1)
        self.assertEqual(games_played("12-4"), 16)

    def test_a_three_part_record_counts_the_draws(self) -> None:
        self.assertEqual(games_played("5-3-1"), 9)

    def test_nothing_played_is_zero_not_an_error(self) -> None:
        for record in ("0-0", "", None, "not a record"):
            with self.subTest(record=record):
                self.assertEqual(games_played(record), 0)


class StrengthShrinkageTests(unittest.TestCase):
    """`n / (n + k)`, the same rule already used for league intercepts."""

    def _strength(self, games=None, home_power=0.8, away_power=0.0):
        features = {"homePower": home_power, "awayPower": away_power}
        if games is not None:
            features["strengthGames"] = games
        return build_feature_dict(features)["strengthDiff"]

    def test_a_preseason_game_contributes_no_strength(self) -> None:
        """Zero games is zero evidence, which is the whole preseason case."""
        self.assertEqual(self._strength(0), 0.0)

    def test_one_game_is_worth_almost_nothing(self) -> None:
        """The live NFL shape: a 0.80 gap off a single opener."""
        self.assertAlmostEqual(self._strength(1), 0.8 * (1 / 11), places=4)
        self.assertLess(self._strength(1), self._strength(None) / 10)

    def test_a_full_season_is_barely_touched(self) -> None:
        """Close to a no-op where records are mature, which is the point."""
        full = self._strength(162)
        self.assertGreater(full / self._strength(None), 0.93)

    def test_weight_rises_with_the_sample(self) -> None:
        values = [self._strength(n) for n in (1, 5, 10, 20, 60, 162)]
        self.assertEqual(values, sorted(values))

    def test_half_weight_lands_at_the_documented_constant(self) -> None:
        self.assertAlmostEqual(
            self._strength(STRENGTH_SHRINK_GAMES), self._strength(None) / 2, places=4
        )

    def test_rows_logged_before_the_fix_are_left_exactly_as_they_were(self) -> None:
        """No strengthGames means no shrinkage, not a guessed one.

        Every row logged before 13 Aug 2026 lacks the field. Imputing a sample
        size for them would silently rewrite the training set.
        """
        self.assertAlmostEqual(self._strength(None), 0.8, places=4)

    def test_it_shrinks_toward_zero_from_both_directions(self) -> None:
        """Away-favoured games must not be pushed the wrong way."""
        self.assertLess(self._strength(1, home_power=0.0, away_power=0.8), 0)
        self.assertGreater(self._strength(1, home_power=0.0, away_power=0.8), -0.1)

    def test_an_absent_strength_stays_absent(self) -> None:
        """Shrinking None must not manufacture a zero, which means something else."""
        self.assertIsNone(build_feature_dict({"strengthGames": 5})["strengthDiff"])


class SeasonTypeTests(unittest.TestCase):
    """ESPN always published it; the parser always threw it away."""

    def _game(self, season_type):
        from espn_client import parse_scoreboard

        payload = {"events": [{
            "id": "1",
            "season": {"type": season_type},
            "competitions": [{
                "date": "2026-08-13T23:00Z",
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": "New England Patriots",
                                                  "abbreviation": "NE"}, "records": []},
                    {"homeAway": "away", "team": {"displayName": "Indianapolis Colts",
                                                  "abbreviation": "IND"}, "records": []},
                ],
                "status": {"type": {"state": "pre", "completed": False}},
            }],
        }]}
        return parse_scoreboard(payload, league="nfl")[0]

    def test_the_parser_keeps_the_season_type(self) -> None:
        self.assertEqual(self._game(1)["seasonType"], 1)
        self.assertEqual(self._game(2)["seasonType"], 2)

    def test_preseason_reports_no_games_behind_its_strength(self) -> None:
        """Starters barely play, so an exhibition result is not evidence about
        the clubs -- and reporting zero lets the shrinkage rule handle it with
        no second code path."""
        from mlb_predictions import extract_model_inputs

        preseason = {"league": "nfl", "seasonType": 1,
                     "homeRecord": "1-0", "awayRecord": "0-1"}
        self.assertEqual(extract_model_inputs(preseason)["strengthGames"], 0)

    def test_the_regular_season_counts_the_thinner_of_the_two_records(self) -> None:
        """A 12-4 club against a 1-0 club is a 1-game question, not a 16."""
        from mlb_predictions import extract_model_inputs

        regular = {"league": "nfl", "seasonType": 2,
                   "homeRecord": "12-4", "awayRecord": "1-0"}
        self.assertEqual(extract_model_inputs(regular)["strengthGames"], 1)

    def test_a_missing_season_type_is_treated_as_a_real_fixture(self) -> None:
        """Absent means unknown, and most leagues here have no exhibitions."""
        from mlb_predictions import extract_model_inputs

        unknown = {"league": "mlb", "homeRecord": "60-40", "awayRecord": "55-45"}
        self.assertEqual(extract_model_inputs(unknown)["strengthGames"], 100)


class DivergenceGateTests(unittest.TestCase):
    """The check that would have caught this without anyone asking."""

    def _report(self, by_league):
        return {"divergence": {"byLeague": by_league}}

    def test_a_league_far_from_the_market_fails_the_build(self) -> None:
        from scripts.check_regression import check_divergence

        failures = check_divergence(self._report({
            "nfl": {"n": 40, "medianGapPct": 27.2},
            "mlb": {"n": 658, "medianGapPct": 8.2},
        }))
        self.assertEqual(len(failures), 1)
        self.assertIn("nfl", failures[0])

    def test_every_working_league_passes(self) -> None:
        """Measured 13 Aug: 4.5 to 8.2. The gate must not nag about these."""
        from scripts.check_regression import check_divergence

        self.assertEqual(check_divergence(self._report({
            "mlb": {"n": 658, "medianGapPct": 8.2},
            "wnba": {"n": 50, "medianGapPct": 4.5},
            "afl": {"n": 40, "medianGapPct": 5.6},
        })), [])

    def test_a_thin_sample_is_exempt(self) -> None:
        """One preseason game can post a 40-point median honestly, and failing
        the build over it would teach everyone to ignore this."""
        from scripts.check_regression import check_divergence

        self.assertEqual(check_divergence(self._report({
            "nfl": {"n": 1, "medianGapPct": 40.5},
        })), [])

    def test_a_report_without_the_block_is_not_a_failure(self) -> None:
        from scripts.check_regression import check_divergence

        self.assertEqual(check_divergence({}), [])

    def test_the_evaluation_actually_emits_the_block(self) -> None:
        """A gate reading a key nothing writes is worse than no gate."""
        source = (ROOT / "scripts" / "evaluation.py").read_text(encoding="utf-8")
        self.assertIn('report["byLeague"]', source)


if __name__ == "__main__":
    unittest.main()
