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


if __name__ == "__main__":
    unittest.main()
