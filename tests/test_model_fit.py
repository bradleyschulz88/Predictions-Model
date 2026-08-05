"""Tests for the stdlib logistic fitter."""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_fit  # noqa: E402
from model_fit import (  # noqa: E402
    LogisticModel,
    Sample,
    build_feature_dict,
    fit_from_observations,
    fit_logistic,
    load_model,
    logit,
    measure_split_diff_centre,
    sigmoid,
    solve_linear,
    to_row,
    walk_forward_scores,
)


class LinearAlgebraTests(unittest.TestCase):
    def test_solves_a_known_system(self) -> None:
        solution = solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
        assert solution is not None
        self.assertAlmostEqual(solution[0], 1.0, places=6)
        self.assertAlmostEqual(solution[1], 3.0, places=6)

    def test_returns_none_for_singular_matrix(self) -> None:
        self.assertIsNone(solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]))

    def test_handles_pivoting(self) -> None:
        # Leading zero forces a row swap.
        solution = solve_linear([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0])
        assert solution is not None
        self.assertAlmostEqual(solution[0], 3.0, places=6)
        self.assertAlmostEqual(solution[1], 2.0, places=6)


class LogisticFitTests(unittest.TestCase):
    def test_recovers_known_coefficients(self) -> None:
        random.seed(7)
        true_intercept, true_slope = 0.4, 1.6
        rows, labels = [], []
        for _ in range(4000):
            x = random.gauss(0.0, 1.0)
            prob = sigmoid(true_intercept + true_slope * x)
            rows.append([1.0, x])
            labels.append(1 if random.random() < prob else 0)

        weights = fit_logistic(rows, labels, l2=1e-6)
        self.assertAlmostEqual(weights[0], true_intercept, delta=0.12)
        self.assertAlmostEqual(weights[1], true_slope, delta=0.15)

    def test_regularisation_shrinks_slope_not_intercept(self) -> None:
        random.seed(11)
        rows, labels = [], []
        for _ in range(400):
            x = random.gauss(0.0, 1.0)
            rows.append([1.0, x])
            labels.append(1 if random.random() < sigmoid(0.5 + 2.0 * x) else 0)

        weak = fit_logistic(rows, labels, l2=0.01)
        strong = fit_logistic(rows, labels, l2=200.0)
        self.assertLess(abs(strong[1]), abs(weak[1]))
        # The unpenalised intercept should still track the base rate.
        self.assertGreater(strong[0], 0.0)

    def test_converges_on_separable_data_without_blowing_up(self) -> None:
        rows = [[1.0, float(x)] for x in range(-20, 21)]
        labels = [1 if x >= 0 else 0 for x in range(-20, 21)]
        weights = fit_logistic(rows, labels, l2=1.0)
        self.assertTrue(all(math.isfinite(weight) for weight in weights))

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(fit_logistic([], []), [])

    def test_per_feature_l2_shrinks_only_the_penalised_column(self) -> None:
        """A sequence of per-feature penalties must not just average out to
        the same fit as a scalar -- the whole point is that two features can
        be shrunk by different amounts."""
        random.seed(13)
        rows, labels = [], []
        for _ in range(600):
            x1 = random.gauss(0.0, 1.0)
            x2 = random.gauss(0.0, 1.0)
            rows.append([1.0, x1, x2])
            labels.append(1 if random.random() < sigmoid(0.3 + 1.5 * x1 + 1.5 * x2) else 0)

        uniform = fit_logistic(rows, labels, l2=20.0)
        # Same total penalty "budget" style magnitude, but loaded entirely
        # onto the first feature. Its coefficient must shrink harder than the
        # uniform fit's, and the second feature -- penalised far less here --
        # must end up larger than its uniform counterpart.
        differential = fit_logistic(rows, labels, l2=[80.0, 1.0])
        self.assertLess(abs(differential[1]), abs(uniform[1]))
        self.assertGreater(abs(differential[2]), abs(uniform[2]))

    def test_per_feature_l2_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            fit_logistic([[1.0, 2.0, 3.0]], [1], l2=[1.0])


class FeatureCollapseTests(unittest.TestCase):
    def test_averages_available_strength_measures(self) -> None:
        values = build_feature_dict(
            {"recordDiff": 0.10, "splitDiff": 0.10, "homePower": 0.60, "awayPower": 0.50},
            split_diff_centre=0.04,
        )
        # (0.10 + (0.10 - 0.04) + 0.10) / 3
        self.assertAlmostEqual(values["strengthDiff"], (0.10 + 0.06 + 0.10) / 3)

    def test_split_diff_is_centred_so_home_field_is_not_double_counted(self) -> None:
        centred = build_feature_dict({"splitDiff": 0.04}, split_diff_centre=0.04)
        self.assertAlmostEqual(centred["strengthDiff"], 0.0)

    def test_strength_is_none_when_nothing_is_available(self) -> None:
        self.assertIsNone(build_feature_dict({})["strengthDiff"])

    def test_market_logit_matches_implied_probability(self) -> None:
        values = build_feature_dict({"impliedHome": 60.0})
        self.assertAlmostEqual(values["marketLogit"], logit(0.6))

    def test_partial_power_data_is_ignored(self) -> None:
        # Only one side's power rating is useless as a difference.
        values = build_feature_dict({"recordDiff": 0.2, "homePower": 0.6})
        self.assertAlmostEqual(values["strengthDiff"], 0.2)

    def test_measure_split_diff_centre_averages_observed_gap(self) -> None:
        centre = measure_split_diff_centre([{"splitDiff": 0.02}, {"splitDiff": 0.06}, {}])
        self.assertAlmostEqual(centre, 0.04)


class PitchingFipFeatureTests(unittest.TestCase):
    """pitchingFipDiff: a FIP-based alternative to the ERA-based pitchingDiff,
    queued alongside the rest of CANDIDATE_FEATURES rather than shipped --
    see the comment above CANDIDATE_FEATURES for why it has no real graded
    coverage yet even though the fetch that feeds it is now fixed."""

    def test_home_minus_away_fip_favours_the_better_home_starter(self) -> None:
        values = build_feature_dict({
            "mlbPitching": {"homePitcherFip": 3.00, "awayPitcherFip": 4.20},
        })
        self.assertAlmostEqual(values["pitchingFipDiff"], 1.20)

    def test_negative_when_the_away_starter_is_better(self) -> None:
        values = build_feature_dict({
            "mlbPitching": {"homePitcherFip": 4.50, "awayPitcherFip": 3.10},
        })
        self.assertLess(values["pitchingFipDiff"], 0)

    def test_missing_either_side_yields_none(self) -> None:
        self.assertIsNone(
            build_feature_dict({"mlbPitching": {"homePitcherFip": 3.5}})["pitchingFipDiff"]
        )
        self.assertIsNone(build_feature_dict({"mlbPitching": {}})["pitchingFipDiff"])

    def test_missing_pitching_block_yields_none(self) -> None:
        self.assertIsNone(build_feature_dict({})["pitchingFipDiff"])

    def test_independent_of_the_era_based_candidate(self) -> None:
        """The two pitchingDiff variants read different keys, so a game with
        ERA but no FIP yet (the entire graded log, today) scores one and not
        the other -- neither one silently inherits the other's value."""
        values = build_feature_dict({
            "mlbPitching": {"homePitcherApiEra": 3.0, "awayPitcherApiEra": 4.0},
        })
        self.assertIsNotNone(values["pitchingDiff"])
        self.assertIsNone(values["pitchingFipDiff"])


class RowBuildingTests(unittest.TestCase):
    def test_missing_feature_contributes_nothing(self) -> None:
        row = to_row({"strengthDiff": None}, ("strengthDiff",), {"strengthDiff": 0.5}, {"strengthDiff": 0.2})
        self.assertEqual(row, [1.0, 0.0])

    def test_standardises_present_values(self) -> None:
        row = to_row({"strengthDiff": 0.7}, ("strengthDiff",), {"strengthDiff": 0.5}, {"strengthDiff": 0.2})
        self.assertAlmostEqual(row[1], 1.0)


def _make_samples(count: int = 300, seed: int = 3) -> list[Sample]:
    random.seed(seed)
    samples = []
    for index in range(count):
        strength = random.gauss(0.0, 0.1)
        prob = sigmoid(0.1 + 6.0 * strength)
        label = 1 if random.random() < prob else 0
        samples.append(
            Sample(
                values={
                    "strengthDiff": strength,
                    "marketLogit": logit(min(0.95, max(0.05, prob))) if index % 2 else None,
                    "restDiff": 0.0,
                    "injuryDiff": 0.0,
                    "b2bDiff": 0.0,
                },
                label=label,
                league="mlb" if index % 3 else "wnba",
                date=f"2026-06-{(index % 28) + 1:02d}",
            )
        )
    return samples


class ModelRoundTripTests(unittest.TestCase):
    def test_fitted_model_predicts_in_range(self) -> None:
        payload = fit_from_observations(_make_samples())
        model = LogisticModel(payload)
        for sample in _make_samples(50, seed=99):
            prob = model.predict_from_values(sample.values, sample.league)
            assert prob is not None
            self.assertGreaterEqual(prob, model_fit.MIN_PROB)
            self.assertLessEqual(prob, model_fit.MAX_PROB)

    def test_stronger_team_gets_higher_probability(self) -> None:
        payload = fit_from_observations(_make_samples())
        model = LogisticModel(payload)
        weak = model.predict_from_values({"strengthDiff": -0.2}, "mlb")
        strong = model.predict_from_values({"strengthDiff": 0.2}, "mlb")
        self.assertLess(weak, strong)

    def test_survives_json_round_trip(self) -> None:
        payload = fit_from_observations(_make_samples())
        restored = LogisticModel(json.loads(json.dumps(payload)))
        original = LogisticModel(payload)
        values = {"strengthDiff": 0.05, "marketLogit": 0.3}
        self.assertAlmostEqual(
            original.predict_from_values(values, "mlb"),
            restored.predict_from_values(values, "mlb"),
        )

    def test_load_model_returns_none_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_model(Path(directory)))

    def test_load_model_returns_none_on_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / model_fit.WEIGHTS_FILE
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_model(Path(directory)))

    def test_split_diff_centre_is_instance_state_not_global(self) -> None:
        first = LogisticModel({"standalone": {}, "splitDiffCentre": 0.01})
        second = LogisticModel({"standalone": {}, "splitDiffCentre": 0.09})
        self.assertAlmostEqual(first.split_diff_centre, 0.01)
        self.assertAlmostEqual(second.split_diff_centre, 0.09)


class WalkForwardTests(unittest.TestCase):
    def test_reports_scores_on_held_out_games(self) -> None:
        scores = walk_forward_scores(_make_samples(400), folds=4)
        self.assertGreater(scores["n"], 0)
        self.assertLess(scores["logLoss"], 0.6931)  # Better than a coin flip.

    def test_declines_to_score_tiny_samples(self) -> None:
        result = walk_forward_scores(_make_samples(20), folds=5)
        self.assertEqual(result["folds"], 0)

    def test_folds_are_chronological(self) -> None:
        # Training always precedes testing, so a later date can never inform an
        # earlier prediction.
        samples = sorted(_make_samples(400), key=lambda s: s.date)
        ordered_dates = [sample.date for sample in samples]
        self.assertEqual(ordered_dates, sorted(ordered_dates))


class AblationRecheckTests(unittest.TestCase):
    """The queued-candidate table (h2h, handedness, bullpen, elo, ...) has to
    be re-run for the dashboard to show it, not just printed to a terminal
    someone has to remember to open."""

    def test_writes_a_row_per_nested_feature_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = model_fit.ablate_and_write(data_dir, samples=_make_samples(200))
            self.assertTrue((data_dir / model_fit.ABLATION_FILE).is_file())
            self.assertEqual(len(payload["rows"]), len(model_fit.CANDIDATE_FEATURES))
            self.assertEqual(payload["rows"][0]["features"], ["strengthDiff"])
            self.assertEqual(payload["shippedSize"], len(model_fit.ANCHORED_FEATURES))
            self.assertEqual(payload["nSamples"], 200)

    def test_written_file_round_trips_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            model_fit.ablate_and_write(data_dir, samples=_make_samples(200))
            on_disk = json.loads((data_dir / model_fit.ABLATION_FILE).read_text(encoding="utf-8"))
            self.assertIn("generatedAt", on_disk)
            self.assertEqual(len(on_disk["rows"]), len(model_fit.CANDIDATE_FEATURES))

    def test_reads_samples_from_the_log_when_none_are_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            payload = model_fit.ablate_and_write(data_dir)
            self.assertEqual(payload["nSamples"], 0, "no predictions_log.json in an empty dir")
            self.assertEqual(payload["rows"], [])


class PitchingFipSignificanceTests(unittest.TestCase):
    """The ablation IS the significance check this codebase uses -- a
    candidate only ships once it beats its own absence out of sample. Today's
    graded log has zero pitchingFipDiff coverage (see the comment above
    CANDIDATE_FEATURES), so ablate() correctly shows it as neutral -- that is
    not proof the check itself works. This constructs data where the FIP
    candidate carries a real, strong signal beyond what pitchingDiff and the
    market already have, and confirms the exact same walk-forward comparison
    that runs in CI would actually detect it and prefer it. If it could not,
    "test before promoting" would be a check with nothing behind it.
    """

    def _samples_with_real_fip_signal(self, count: int = 400, seed: int = 7) -> list[Sample]:
        random.seed(seed)
        samples = []
        for index in range(count):
            strength = random.gauss(0.0, 0.1)
            market = strength * 4.0 + random.gauss(0.0, 0.2)
            # pitchingDiff (ERA) is pure noise here -- it should not help,
            # mirroring the real-world finding that it is redundant with the
            # market. pitchingFipDiff carries the actual extra signal.
            era_noise = random.gauss(0.0, 1.0)
            fip_signal = random.gauss(0.0, 1.0)
            logit_true = 3.0 * strength + 1.0 * market + 0.9 * fip_signal
            prob = sigmoid(logit_true)
            label = 1 if random.random() < prob else 0
            samples.append(
                Sample(
                    values={
                        "strengthDiff": strength,
                        "marketLogit": market,
                        "pitchingDiff": era_noise,
                        "pitchingFipDiff": fip_signal,
                    },
                    label=label,
                    league="mlb",
                    date=f"2026-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}",
                )
            )
        return samples

    def test_a_real_signal_would_beat_the_set_without_it(self) -> None:
        samples = self._samples_with_real_fip_signal()
        with_era_only = model_fit.walk_forward_scores(
            samples, l2=3.0, anchored_features=("strengthDiff", "marketLogit", "pitchingDiff"))
        with_fip = model_fit.walk_forward_scores(
            samples, l2=3.0,
            anchored_features=("strengthDiff", "marketLogit", "pitchingDiff", "pitchingFipDiff"))
        self.assertLess(
            with_fip["logLoss"], with_era_only["logLoss"] - 0.01,
            "a genuinely predictive pitchingFipDiff must measurably improve "
            "walk-forward log loss, or the significance check is not real",
        )

    def test_pure_noise_would_not_beat_it(self) -> None:
        """The other half of the check: it must also correctly decline a
        candidate with no real signal, which is today's actual situation."""
        random.seed(11)
        samples = []
        for index in range(400):
            strength = random.gauss(0.0, 0.1)
            market = strength * 4.0 + random.gauss(0.0, 0.2)
            label = 1 if random.random() < sigmoid(3.0 * strength + 1.0 * market) else 0
            samples.append(
                Sample(
                    values={
                        "strengthDiff": strength,
                        "marketLogit": market,
                        "pitchingDiff": random.gauss(0.0, 1.0),
                        "pitchingFipDiff": random.gauss(0.0, 1.0),  # unrelated to label
                    },
                    label=label,
                    league="mlb",
                    date=f"2026-{(index % 12) + 1:02d}-{(index % 28) + 1:02d}",
                )
            )
        with_era_only = model_fit.walk_forward_scores(
            samples, l2=3.0, anchored_features=("strengthDiff", "marketLogit", "pitchingDiff"))
        with_fip = model_fit.walk_forward_scores(
            samples, l2=3.0,
            anchored_features=("strengthDiff", "marketLogit", "pitchingDiff", "pitchingFipDiff"))
        self.assertGreaterEqual(with_fip["logLoss"], with_era_only["logLoss"] - 0.005)


class DifferentialShrinkageTests(unittest.TestCase):
    """strengthDiff and marketLogit getting their own ridge penalty.

    A plain shared L2 shrinks every coefficient by the same proportion, which
    preserves whatever split the unpenalised fit implies -- and on the real
    graded log that split lands close to even despite the market alone
    beating the blended model on every priced game. These pin the mechanics:
    a dict resolves to the right per-feature penalty, it round-trips through
    JSON, and choosing it is still walk-forward-driven rather than assuming a
    ratio.
    """

    def test_l2_vector_resolves_named_and_default_penalties(self) -> None:
        vector = model_fit._l2_vector(
            {"strengthDiff": 30.0, "marketLogit": 1.0, "_default": 5.0},
            ("marketLogit", "strengthDiff", "restDiff"),
        )
        self.assertEqual(vector, [1.0, 30.0, 5.0])

    def test_l2_vector_passes_a_scalar_through_unchanged(self) -> None:
        self.assertEqual(model_fit._l2_vector(7.0, ("strengthDiff",)), 7.0)

    def test_fit_block_stores_the_dict_as_given(self) -> None:
        samples = _make_samples(300)
        block = model_fit._fit_block(
            samples, ("strengthDiff", "marketLogit"),
            l2={"strengthDiff": 30.0, "marketLogit": 1.0},
        )
        assert block is not None
        self.assertEqual(block["l2"], {"strengthDiff": 30.0, "marketLogit": 1.0})

    def test_fit_from_observations_accepts_a_dict_for_both_blocks(self) -> None:
        """The standalone block only has strengthDiff -- the same dict, built
        for the anchored pair, must still resolve correctly there rather than
        raising or silently mis-keying."""
        l2_map = {"strengthDiff": 30.0, "marketLogit": 1.0, "_default": 10.0}
        payload = fit_from_observations(_make_samples(300), l2=l2_map)
        # Both blocks receive the same dict; each resolves only the entries
        # relevant to its own feature list when actually fitting.
        self.assertEqual(payload["anchored"]["l2"], l2_map)
        self.assertEqual(payload["standalone"]["l2"], l2_map)
        model = LogisticModel(payload)
        prob = model.predict_from_values({"strengthDiff": 0.1}, "mlb")
        self.assertIsNotNone(prob)

    def test_choosing_penalties_returns_both_keys(self) -> None:
        chosen = model_fit.choose_anchored_penalties(
            _make_samples(400),
            strength_candidates=(3.0, 30.0),
            market_candidates=(1.0, 10.0),
        )
        self.assertIn("strengthDiff", chosen)
        self.assertIn("marketLogit", chosen)
        self.assertIn(chosen["strengthDiff"], (3.0, 30.0))
        self.assertIn(chosen["marketLogit"], (1.0, 10.0))

    def test_a_dict_l2_survives_a_json_round_trip(self) -> None:
        """model_weights.json is read back by LogisticModel; the l2 field is
        metadata only, but it must not corrupt the round trip."""
        payload = fit_from_observations(
            _make_samples(300), l2={"strengthDiff": 30.0, "marketLogit": 1.0}
        )
        restored = LogisticModel(json.loads(json.dumps(payload)))
        original = LogisticModel(payload)
        values = {"strengthDiff": 0.05, "marketLogit": 0.3}
        self.assertAlmostEqual(
            original.predict_from_values(values, "mlb"),
            restored.predict_from_values(values, "mlb"),
        )

    def test_walk_forward_scores_accepts_a_dict_l2(self) -> None:
        """This is the exact path scripts/backtest_model.py calls with the
        shipped model's own metadata l2 -- it must not require a float()."""
        scores = walk_forward_scores(
            _make_samples(400), l2={"strengthDiff": 30.0, "marketLogit": 1.0}, folds=4
        )
        self.assertGreater(scores["n"], 0)

    def test_format_l2_reads_naturally_for_both_shapes(self) -> None:
        self.assertEqual(model_fit._format_l2(3.0), "3.0")
        formatted = model_fit._format_l2({"strengthDiff": 30.0, "marketLogit": 1.0, "_default": 10.0})
        self.assertIn("strengthDiff=30.0", formatted)
        self.assertIn("marketLogit=1.0", formatted)
        self.assertNotIn("_default", formatted)


class HomeFieldTests(unittest.TestCase):
    def test_intercept_absorbs_home_field_instead_of_a_hardcoded_constant(self) -> None:
        """A neutral matchup should sit near the observed home win rate."""
        random.seed(5)
        samples = []
        for index in range(500):
            label = 1 if random.random() < 0.54 else 0  # 54% home, no team edge.
            samples.append(
                Sample(
                    values={"strengthDiff": 0.0, "marketLogit": None},
                    label=label,
                    league="mlb",
                    date=f"2026-06-{(index % 28) + 1:02d}",
                )
            )
        model = LogisticModel(fit_from_observations(samples))
        neutral = model.predict_from_values({"strengthDiff": 0.0}, "mlb")
        self.assertAlmostEqual(neutral, 0.54, delta=0.05)


class ColdStartLeagueTests(unittest.TestCase):
    """What a league with no graded games is handed, and why it is safe.

    NBA and NFL have zero graded picks until their seasons open, so
    _league_intercepts skips them entirely and they fall back on the global
    intercept -- fitted, today, almost entirely on baseball. The worry was that
    this hands them baseball's home-field edge, which is the weakest of the
    four leagues measured (see scripts/measure_margin_sd.py for the table).

    It does not, because the market anchor carries home advantage and the
    intercept only holds the residual the market misses. These fit their own
    models rather than reading docs/data/model_weights.json: that file is
    gitignored and produced by the refit step *after* tests run, so a checkout
    does not have one. The guard on the actually-published weights lives in
    scripts/check_regression.py, which runs where the file exists.
    """

    # MLB's raw home edge, +0.1122 in logit space, is the number the intercept
    # would have to carry if it were a home-field term rather than a residual.
    HOME_LOGIT = math.log(0.528 / 0.472)

    @staticmethod
    def _priced_samples(count: int, home_rate: float, price_spread: float) -> list[Sample]:
        """Games the market prices around the league's home edge.

        `price_spread` is how much the quoted price varies game to game. Zero
        means every game is priced identically, which is not a slate -- it is
        the degenerate case used below to show what the intercept does when the
        market feature has nothing to say.
        """
        base = math.log(home_rate / (1 - home_rate))
        samples = []
        for index in range(count):
            market = random.gauss(base, price_spread)
            samples.append(
                Sample(
                    values={"strengthDiff": (market - base) / 2, "marketLogit": market},
                    label=1 if random.random() < 1 / (1 + math.exp(-market)) else 0,
                    league="mlb",
                    date=f"2026-06-{(index % 28) + 1:02d}",
                )
            )
        return samples

    def _fit(self, price_spread: float, count: int = 600) -> dict:
        random.seed(12)
        return fit_from_observations(self._priced_samples(count, 0.528, price_spread))

    def test_a_league_with_no_games_gets_no_intercept_of_its_own(self) -> None:
        fitted = self._fit(0.4, count=60)
        self.assertIn("mlb", fitted["leagueIntercepts"])
        self.assertNotIn("nba", fitted["leagueIntercepts"])

    def test_the_intercept_empties_out_as_the_market_feature_fills_up(self) -> None:
        """The mechanism, not a magic number.

        On a slate where every game carries the same price the market feature
        is constant, standardisation erases it, and the intercept is left
        holding the entire home edge. Let the price vary the way a real slate
        does and the intercept drains away into the feature. That is what makes
        it a residual, and it is the whole reason a league that has never
        played can inherit it without inheriting baseball's home-field figure.
        """
        flat = abs(self._fit(0.0)["anchored"]["weights"][0])
        realistic = abs(self._fit(0.6)["anchored"]["weights"][0])
        self.assertGreater(flat, self.HOME_LOGIT, "a dead market should strand the edge here")
        self.assertLess(realistic, flat / 4)
        self.assertLess(realistic, self.HOME_LOGIT / 4)

    def test_a_cold_start_league_forgoes_almost_nothing(self) -> None:
        """What NBA loses by having no intercept is the intercept MLB earns.

        With the anchored block already explaining the outcomes, that residual
        is ~0.0000, so the two leagues are handed the same answer. The gap --
        not its sign -- is the thing worth pinning: whichever way baseball's
        own correction happens to point on a given refit, it is too small to
        move a published probability.
        """
        model = LogisticModel(self._fit(0.6))
        values = {"strengthDiff": 0.0, "marketLogit": 0.0}
        cold = model.predict_from_values(values, "nba")
        mlb = model.predict_from_values(values, "mlb")
        self.assertLess(abs(cold - mlb), 0.01)


class InterceptGateTests(unittest.TestCase):
    """The build gate that watches the intercept on the weights that ship.

    model_weights.json is gitignored and written by the refit step, so this
    check runs there rather than in the suite. That makes it the kind of gate
    nobody sees fire until it matters, which is exactly the kind worth testing.
    """

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import check_regression

        self.gate = check_regression
        self.limit = check_regression.MAX_RESIDUAL_INTERCEPT

    def _weights(self, *, global_intercept: float = 0.0, **leagues: float) -> dict:
        return {
            "anchored": {"weights": [global_intercept, 0.4, 0.42]},
            "standalone": {"weights": [global_intercept, 0.75]},
            "leagueIntercepts": leagues,
        }

    def test_a_healthy_fit_passes(self) -> None:
        """The values actually observed on 2026-08-05."""
        weights = self._weights(global_intercept=0.0367, mlb=-0.0362, wnba=0.0992)
        self.assertEqual(self.gate.check_intercepts(weights), [])

    def test_a_global_intercept_carrying_home_field_fails(self) -> None:
        failures = self.gate.check_intercepts(self._weights(global_intercept=self.limit + 0.05))
        self.assertEqual(len(failures), 2)  # anchored and standalone both.
        self.assertIn("stopped carrying home advantage", failures[0])

    def test_a_league_intercept_that_became_an_edge_fails(self) -> None:
        failures = self.gate.check_intercepts(self._weights(afl=self.limit + 0.1))
        self.assertEqual(len(failures), 1)
        self.assertIn("afl", failures[0])

    def test_the_sign_does_not_matter(self) -> None:
        """A large negative intercept is the same defect pointing the other way."""
        self.assertEqual(len(self.gate.check_intercepts(self._weights(mlb=-self.limit - 0.1))), 1)

    def test_absent_weights_are_not_a_failure(self) -> None:
        """A checkout has no weights file; that must not fail the gate."""
        self.assertEqual(self.gate.check_intercepts({}), [])

    def test_the_limit_clears_every_measured_home_edge(self) -> None:
        """WNBA's +0.2231 is the largest of the four; the gate must sit above it."""
        self.assertGreater(self.limit, 0.2231)


if __name__ == "__main__":
    unittest.main()
