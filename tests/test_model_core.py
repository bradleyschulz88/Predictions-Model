"""Tests for probability resolution and the fitted/fallback split."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_core  # noqa: E402
from model_core import resolve_probabilities  # noqa: E402


@dataclass
class FakeLeague:
    id: str = "mlb"
    supports_draw: bool = False


class FakeModel:
    """Stand-in for a fitted LogisticModel with a fixed answer."""

    def __init__(self, probability: float | None) -> None:
        self.probability = probability
        self.metadata: dict = {}

    def predict_proba(self, features, league):  # noqa: ANN001
        return self.probability


def _resolve(**overrides):
    defaults = dict(
        game={"league": "mlb"},
        model_inputs={"recordDiff": 0.1},
        heuristic_home=0.85,
        enrichment={},
        league_config=FakeLeague(),
        league="mlb",
        # Identity calibration unless a test overrides it.
        legacy_calibrate=lambda prob, **_: prob,
    )
    defaults.update(overrides)
    return resolve_probabilities(**defaults)


class FittedPathTests(unittest.TestCase):
    def setUp(self) -> None:
        model_core.reset_fitted_model_cache()

    def tearDown(self) -> None:
        model_core.reset_fitted_model_cache()

    def test_uses_fitted_probability_when_available(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=FakeModel(0.62)):
            result = _resolve()
        self.assertEqual(result["method"], "fitted")
        self.assertAlmostEqual(result["home"], 0.62)

    def test_fitted_probability_is_not_shrunk_again(self) -> None:
        """Shrinking a maximum-likelihood estimate would decalibrate it."""
        calibrate = mock.Mock(side_effect=lambda prob, **_: 0.5)
        with mock.patch.object(model_core, "load_model", return_value=FakeModel(0.62)):
            result = _resolve(legacy_calibrate=calibrate)
        calibrate.assert_not_called()
        self.assertAlmostEqual(result["home"], 0.62)

    def test_falls_back_when_model_declines_to_predict(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=FakeModel(None)):
            result = _resolve()
        self.assertEqual(result["method"], "heuristic")

    def test_falls_back_when_no_weights_file(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=None):
            result = _resolve()
        self.assertEqual(result["method"], "heuristic")
        self.assertFalse(result["fittedAvailable"])

    def test_heuristic_path_applies_calibration(self) -> None:
        calibrate = mock.Mock(side_effect=lambda prob, **_: 0.55)
        with mock.patch.object(model_core, "load_model", return_value=None):
            result = _resolve(legacy_calibrate=calibrate)
        calibrate.assert_called_once()
        self.assertAlmostEqual(result["home"], 0.55)

    def test_model_is_loaded_once_per_process(self) -> None:
        loader = mock.Mock(return_value=FakeModel(0.6))
        with mock.patch.object(model_core, "load_model", loader):
            _resolve()
            _resolve()
        self.assertEqual(loader.call_count, 1)


class ProbabilityShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        model_core.reset_fitted_model_cache()

    def tearDown(self) -> None:
        model_core.reset_fitted_model_cache()

    def test_two_way_probabilities_sum_to_one(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=FakeModel(0.62)):
            result = _resolve()
        self.assertAlmostEqual(result["home"] + result["away"], 1.0)

    def test_probabilities_stay_inside_the_clamp(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=None):
            extreme = _resolve(heuristic_home=0.999)
        self.assertLessEqual(extreme["home"], model_core.MAX_PROB)
        self.assertGreaterEqual(extreme["away"], model_core.MIN_PROB)


class DrawHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        model_core.reset_fitted_model_cache()

    def tearDown(self) -> None:
        model_core.reset_fitted_model_cache()

    def _soccer(self, probability: float = 0.60):
        with mock.patch.object(model_core, "load_model", return_value=FakeModel(probability)):
            return _resolve(
                league_config=FakeLeague(id="epl", supports_draw=True),
                league="epl",
                enrichment={"leagueMetrics": {"drawBaseRate": 0.25}},
            )

    def test_three_way_probabilities_sum_to_one(self) -> None:
        result = self._soccer()
        total = result["home"] + result["away"] + result["draw"]
        self.assertAlmostEqual(total, 1.0)

    def test_draw_is_not_pulled_toward_fifty_percent(self) -> None:
        """The old path shrank the draw toward 0.5, inflating it on every build."""
        result = self._soccer()
        self.assertLess(result["draw"], 0.35)
        self.assertGreater(result["draw"], 0.05)

    def test_draw_probability_respects_its_bounds(self) -> None:
        for probability in (0.10, 0.50, 0.90):
            result = self._soccer(probability)
            self.assertGreaterEqual(result["draw"], 0.08)
            self.assertLessEqual(result["draw"], 0.32)

    def test_favourite_still_leads_after_the_draw_split(self) -> None:
        result = self._soccer(0.70)
        self.assertGreater(result["home"], result["away"])

    def test_binary_probability_is_preserved_for_reporting(self) -> None:
        result = self._soccer(0.70)
        self.assertAlmostEqual(result["binaryHome"], 0.70)


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        model_core.reset_fitted_model_cache()

    def tearDown(self) -> None:
        model_core.reset_fitted_model_cache()

    def test_reports_heuristic_when_unfitted(self) -> None:
        with mock.patch.object(model_core, "load_model", return_value=None):
            self.assertEqual(model_core.model_metadata()["method"], "heuristic")


if __name__ == "__main__":
    unittest.main()
