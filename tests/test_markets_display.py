"""Totals, runline and conditions -- the markets that never reached the card."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mlb_predictions  # noqa: E402
from mlb_predictions import (  # noqa: E402
    RUNLINE_COVER_GIVEN_WIN,
    apply_predictions,
    predict_runline,
    run_environment,
)

APP_JS = ROOT / "dashboard" / "app.js"

LINES = [
    {"sportsbook": "DK", "viewType": "MoneyLine", "currentLine": {"home": -150, "away": 130}},
    {"sportsbook": "DK", "viewType": "Total", "currentLine": {"over": 8.5, "under": 8.5}},
    {"sportsbook": "DK", "viewType": "Spread", "currentLine": {"home": -1.5, "away": 1.5}},
]


def _game(home="Colorado Rockies", away="San Diego Padres", lines=None, venue="Coors Field"):
    return {
        "eventId": "1",
        "league": "mlb",
        "homeTeam": home,
        "awayTeam": away,
        "homeRecord": "55-45",
        "awayRecord": "45-55",
        "venueName": venue,
        "lines": LINES if lines is None else lines,
    }


def _predict(games):
    with patch.object(mlb_predictions, "enrich_games_with_providers", lambda *a, **k: None):
        apply_predictions(games)
    return games


class RunEnvironmentTests(unittest.TestCase):
    """Ballpark plus forecast, as an over/under percentage."""

    def test_shown_without_any_market(self) -> None:
        """The whole point: conditions do not need a posted total to be known."""
        game = _predict([_game(lines=[])])[0]
        environment = game["prediction"]["runEnvironment"]
        self.assertIsNotNone(environment)
        self.assertIsNone(game["prediction"].get("total"), "no market means no totals pick")

    def test_hitters_park_leans_over_and_pitchers_park_under(self) -> None:
        coors = _predict([_game()])[0]["prediction"]["runEnvironment"]
        safeco = _predict([_game("Seattle Mariners", "Texas Rangers", venue="T-Mobile Park")])[0][
            "prediction"
        ]["runEnvironment"]
        self.assertEqual(coors["lean"], "over")
        self.assertEqual(safeco["lean"], "under")
        self.assertGreater(coors["overPct"], safeco["overPct"])

    def test_percentages_are_complementary(self) -> None:
        environment = _predict([_game()])[0]["prediction"]["runEnvironment"]
        self.assertAlmostEqual(environment["overPct"] + environment["underPct"], 100.0, places=1)

    def test_neutral_park_says_neutral(self) -> None:
        game = _predict([_game("Kansas City Royals", "Detroit Tigers", venue="Kauffman Stadium")])[0]
        self.assertEqual(game["prediction"]["runEnvironment"]["lean"], "neutral")

    def test_flagged_as_ungraded(self) -> None:
        """It has never been scored against totals results, so it must say so."""
        self.assertTrue(_predict([_game()])[0]["prediction"]["runEnvironment"]["unvalidated"])

    def test_not_produced_outside_baseball(self) -> None:
        self.assertIsNone(run_environment({"league": "nba", "homeTeam": "Colorado Rockies"}, {}))


class RunlineTests(unittest.TestCase):
    """Baseball's handicap is fixed at +/-1.5, so it is not a spread."""

    def test_mlb_games_now_get_a_handicap(self) -> None:
        """No MLB game ever showed one: MARGIN_STD_DEV had no mlb entry."""
        spread = _predict([_game()])[0]["prediction"].get("spread")
        self.assertIsNotNone(spread)
        self.assertEqual(spread["market"], "runline")

    def test_cover_probability_follows_the_measured_rate(self) -> None:
        prediction = {"probabilities": {"true": {"home": 0.60, "away": 0.40}}}
        result = predict_runline(_game(), LINES, prediction)
        self.assertAlmostEqual(result["homePct"], 60.0 * RUNLINE_COVER_GIVEN_WIN, places=1)

    def test_cover_percentages_are_complementary(self) -> None:
        prediction = {"probabilities": {"true": {"home": 0.62, "away": 0.38}}}
        result = predict_runline(_game(), LINES, prediction)
        self.assertAlmostEqual(result["homePct"] + result["awayPct"], 100.0, places=1)

    def test_a_favourite_is_less_likely_to_cover_than_to_win(self) -> None:
        """The reason a runline pays more than the moneyline."""
        prediction = {"probabilities": {"true": {"home": 0.70, "away": 0.30}}}
        result = predict_runline(_game(), LINES, prediction)
        self.assertLess(result["homePct"], 70.0)

    def test_carries_no_invented_confidence(self) -> None:
        result = predict_runline(_game(), LINES, {"probabilities": {"true": {"home": 0.6, "away": 0.4}}})
        self.assertIsNone(result["confidence"])
        self.assertTrue(result["unvalidated"])

    def test_not_applied_outside_baseball(self) -> None:
        prediction = {"probabilities": {"true": {"home": 0.6, "away": 0.4}}}
        self.assertIsNone(predict_runline({"league": "nba"}, LINES, prediction))

    def test_no_market_means_no_runline(self) -> None:
        prediction = {"probabilities": {"true": {"home": 0.6, "away": 0.4}}}
        self.assertIsNone(predict_runline(_game(), [], prediction))


class CardRenderingTests(unittest.TestCase):
    """These were rendered only inside the collapsed panel, or not at all."""

    def _app_js(self) -> str:
        return APP_JS.read_text(encoding="utf-8")

    def test_conditions_chip_reaches_the_summary_bar(self) -> None:
        self.assertIn("renderRunEnvironmentChip(prediction)", self._app_js())

    def test_totals_chip_shows_a_percentage(self) -> None:
        """"As a percentage" was the request; the chip showed only a pick."""
        self.assertIn("prediction.total.confidence", self._app_js())

    def test_spread_panel_handles_both_shapes(self) -> None:
        """A runline has no modelLine or edgePoints to render."""
        app_js = self._app_js()
        self.assertIn('spread?.market === "runline"', app_js)
        self.assertIn("function spreadChipTitle(spread)", app_js)

    def test_head_to_head_is_rendered(self) -> None:
        app_js = self._app_js()
        self.assertIn("function renderHeadToHead(game)", app_js)
        self.assertIn("${renderHeadToHead(game)}", app_js)

    def test_ungraded_markets_are_labelled_as_such(self) -> None:
        """None of these have been graded; none may read as a calibrated pick."""
        self.assertIn("not graded", self._app_js())


class HeadToHeadFeatureTests(unittest.TestCase):
    def test_logged_as_a_candidate(self) -> None:
        from mlb_predictions import extract_model_inputs

        game = _game()
        game["enrichment"] = {"headToHead": {"homeSeriesWinPct": 0.75, "awaySeriesWinPct": 0.25}}
        self.assertAlmostEqual(extract_model_inputs(game)["h2hDiff"], 0.5)

    def test_absent_series_logs_none_not_zero(self) -> None:
        from mlb_predictions import extract_model_inputs

        self.assertIsNone(extract_model_inputs(_game())["h2hDiff"])

    def test_park_edge_is_logged_too(self) -> None:
        from mlb_predictions import extract_model_inputs

        self.assertEqual(extract_model_inputs(_game())["parkEdge"], 15.0)


if __name__ == "__main__":
    unittest.main()
