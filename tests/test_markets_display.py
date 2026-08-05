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

    def test_confidence_is_the_picked_sides_cover_chance(self) -> None:
        """Derived, not invented -- and not left blank either.

        This was None, which meant a runline could never be priced and so could
        never be ranked against the moneyline. The number was already being
        computed for homePct/awayPct and quoted in the detail text; it just was
        not exposed. It stays `unvalidated` because it comes from the measured
        70.8% clear rate rather than from calibration against its own record.
        """
        result = predict_runline(_game(), LINES, {"probabilities": {"true": {"home": 0.6, "away": 0.4}}})
        self.assertAlmostEqual(
            result["confidence"], max(result["homePct"], result["awayPct"])
        )
        self.assertTrue(result["unvalidated"])

    def test_runline_carries_a_price_so_it_can_be_ranked(self) -> None:
        """MLB runlines go through predict_runline rather than predict_spread,
        so with no odds here every MLB runline was unpriced -- 47 of the 56
        graded spread picks, which left that market's ROI resting on nine."""
        priced = [{"viewType": "Spread", "currentLine": {"home": "-1.5 (+105)", "away": "+1.5 (-125)"}}]
        result = predict_runline(_game(), priced, {"probabilities": {"true": {"home": 0.7, "away": 0.3}}})
        self.assertEqual(result["odds"], 105 if result["pickSide"] == "home" else -125)

    def test_an_sbr_line_still_has_no_price(self) -> None:
        bare = [{"viewType": "Spread", "currentLine": {"home": "-1.5", "away": "+1.5"}}]
        result = predict_runline(_game(), bare, {"probabilities": {"true": {"home": 0.7, "away": 0.3}}})
        self.assertIsNone(result["odds"])

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


class RunlineCoherenceTests(unittest.TestCase):
    """The number attached to a pick must match whether that side is favoured.

    Deriving it from the picked side alone printed "Padres -1.5" for a side the
    model had at 40% -- the opposite bet to the one intended. Four of six sample
    probabilities were wrong.
    """

    LINES = [{"viewType": "Spread", "currentLine": {"home": -1.5, "away": 1.5}}]

    def _runline(self, p_home: float) -> dict:
        from mlb_predictions import predict_runline

        return predict_runline(
            {"league": "mlb", "homeTeam": "Dodgers", "awayTeam": "Padres"},
            self.LINES,
            {"probabilities": {"true": {"home": p_home, "away": 1 - p_home}}},
        )

    def test_the_favourite_lays_and_the_underdog_takes(self) -> None:
        for p_home in (0.20, 0.30, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90):
            result = self._runline(p_home)
            favourite = "home" if p_home >= 0.5 else "away"
            lays = "-1.5" in result["pick"]
            self.assertEqual(
                lays,
                result["pickSide"] == favourite,
                f"p(home)={p_home}: picked {result['pickSide']}, favourite {favourite}, "
                f"pick text {result['pick']!r}",
            )

    def test_an_underdog_pick_takes_the_plus_number(self) -> None:
        """The specific regression: home favoured, away picked."""
        result = self._runline(0.60)
        self.assertEqual(result["pickSide"], "away")
        self.assertIn("+1.5", result["pick"])

    def test_the_stored_line_is_always_the_home_number(self) -> None:
        """Grading reads `line` as the home side's, so the sign must follow."""
        self.assertEqual(self._runline(0.70)["line"], -1.5)
        self.assertEqual(self._runline(0.30)["line"], 1.5)

    def test_the_pick_side_is_the_better_cover_probability(self) -> None:
        for p_home in (0.25, 0.5, 0.75):
            result = self._runline(p_home)
            better = "home" if result["homePct"] >= result["awayPct"] else "away"
            self.assertEqual(result["pickSide"], better)


class GradingRobustnessTests(unittest.TestCase):
    """One malformed row must not abort a grading run over live data."""

    def test_non_finite_scores_are_survivable(self) -> None:
        from accuracy_tracker import grade_spread, grade_total

        for score in (float("inf"), float("-inf"), float("nan")):
            self.assertIsNone(grade_total({"line": 8.5, "pickSide": "over"}, score, 4))
            self.assertIsNone(grade_spread({"line": -1.5, "pickSide": "home"}, score, 4))


class BestBetSelectionTests(unittest.TestCase):
    """Which of the three markets to actually back.

    The card used to give one recommendation -- the moneyline -- and render the
    total and spread as inert text with no pick, price or edge, so the question
    "is the moneyline even the best bet here" had no answer on the page.
    Ranking them needs every market priced through the same assess_price, since
    percentage points of edge are no more comparable across markets than across
    prices.

    The ranking is gated rather than a bare argmax. The moneyline is fitted and
    calibrated against every graded game; the side markets are heuristics that
    have only just started carrying prices. An unvalidated market may show a
    bigger edge, but it may not headline on the strength of it.
    """

    def _prediction(self, *, ml_ev, total_ev=None, spread_ev=None):
        """A prediction whose markets carry exactly the EVs asked for."""
        def block(ev, odds=-110):
            return {"pick": "X", "confidence": 60.0, "odds": odds,
                    "value": {"evPct": ev, "odds": odds, "kellyPct": 1.0, "breakEvenPct": 52.4}}
        prediction = {"outcomeLabel": "Home to win", "confidence": 70.0,
                      "value": {"evPct": ml_ev, "odds": -150, "kellyPct": 2.0, "breakEvenPct": 60.0}}
        if total_ev is not None:
            prediction["total"] = block(total_ev)
        if spread_ev is not None:
            prediction["spread"] = block(spread_ev)
        return prediction

    def _validated(self, **flags):
        """Patch the gate so these tests pin the ranking logic, not the day's
        graded record -- which moves every build."""
        return patch.object(
            mlb_predictions, "market_is_validated",
            side_effect=lambda market: flags.get(market, market == "moneyline"),
        )

    def test_no_priced_market_means_no_best_bet(self) -> None:
        self.assertIsNone(mlb_predictions.select_best_bet({"confidence": 70.0}))

    def test_every_priced_market_is_ranked_by_ev(self) -> None:
        prediction = self._prediction(ml_ev=4.0, total_ev=9.0, spread_ev=-2.0)
        with self._validated(total=True, spread=True):
            best = mlb_predictions.select_best_bet(prediction)
        self.assertEqual([o["market"] for o in best["options"]], ["total", "moneyline", "spread"])

    def test_a_validated_side_market_can_win_the_headline(self) -> None:
        """The gate is not an off switch: a market that has earned it wins."""
        prediction = self._prediction(ml_ev=4.0, total_ev=9.0)
        with self._validated(total=True):
            best = mlb_predictions.select_best_bet(prediction)
        self.assertEqual(best["pick"]["market"], "total")
        self.assertIsNone(best["heldBack"])

    def test_an_unvalidated_market_cannot_headline_on_a_bigger_edge(self) -> None:
        prediction = self._prediction(ml_ev=4.0, spread_ev=30.0)
        with self._validated(spread=False):
            best = mlb_predictions.select_best_bet(prediction)
        self.assertEqual(best["pick"]["market"], "moneyline")
        self.assertEqual(best["heldBack"]["market"], "spread",
                         "the bigger edge must still be reported, not silently dropped")

    def test_a_held_back_market_is_still_ranked_and_visible(self) -> None:
        """Hiding it would be its own dishonesty -- the edge is real data."""
        prediction = self._prediction(ml_ev=4.0, spread_ev=30.0)
        with self._validated(spread=False):
            best = mlb_predictions.select_best_bet(prediction)
        spread = next(o for o in best["options"] if o["market"] == "spread")
        self.assertEqual(spread["evPct"], 30.0)
        self.assertFalse(spread["validated"])

    def test_negative_ev_never_headlines(self) -> None:
        prediction = self._prediction(ml_ev=-3.0, total_ev=-8.0)
        with self._validated(total=True):
            best = mlb_predictions.select_best_bet(prediction)
        self.assertIsNone(best["pick"], "a losing bet is not a recommendation")
        self.assertEqual(len(best["options"]), 2, "but both are still shown")

    def test_the_moneyline_is_always_eligible(self) -> None:
        self.assertTrue(mlb_predictions.market_is_validated("moneyline"))

    def test_a_side_market_needs_priced_history_not_merely_graded(self) -> None:
        """An unpriced pick returns nothing, so it is no evidence that betting
        the market pays -- only the priced count can open the gate."""
        report = {"summary": {"spreads": {"graded": 500, "priced": 9}}}
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
            self.assertEqual(mlb_predictions.market_priced_history("spread"), 9)
            self.assertFalse(mlb_predictions.market_is_validated("spread"))

    def test_enough_priced_history_and_a_winning_record_opens_the_gate(self) -> None:
        report = {"summary": {"totals": {
            "graded": 60, "priced": mlb_predictions.MIN_MARKET_HISTORY,
            "pct": 61.6, "breakEvenPct": 52.3,
        }}}
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
            self.assertTrue(mlb_predictions.market_is_validated("total"))

    def test_a_losing_record_stays_gated_however_much_history_it_has(self) -> None:
        """Priced volume alone is not evidence the market pays. A market that
        has hit under the break-even its own prices imply has shown the
        opposite, and no amount of it should promote the market."""
        report = {"summary": {"totals": {
            "graded": 900, "priced": 800, "pct": 48.0, "breakEvenPct": 52.3,
        }}}
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
            self.assertFalse(mlb_predictions.market_is_validated("total"))

    def test_the_break_even_bar_is_the_one_the_prices_imply(self) -> None:
        """Not a hardcoded 52.4%. The same 55% hit rate passes against -110
        runline-style pricing and fails against a bar the real prices set
        higher, which is what happens once MLB runlines carry odds."""
        cheap = {"summary": {"totals": {"graded": 90, "priced": 90, "pct": 55.0, "breakEvenPct": 52.4}}}
        dear = {"summary": {"totals": {"graded": 90, "priced": 90, "pct": 55.0, "breakEvenPct": 58.0}}}
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=cheap):
            self.assertTrue(mlb_predictions.market_is_validated("total"))
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=dear):
            self.assertFalse(mlb_predictions.market_is_validated("total"))

    def test_a_record_with_no_hit_rate_yet_stays_gated(self) -> None:
        report = {"summary": {"totals": {"graded": 60, "priced": 60, "pct": None}}}
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
            self.assertFalse(mlb_predictions.market_is_validated("total"))

    def test_the_options_carry_the_record_so_the_card_can_caveat_it(self) -> None:
        """A recommended side market has to be able to show its error bar."""
        report = {"summary": {"totals": {
            "graded": 75, "decided": 73, "priced": 64, "pct": 61.6,
            "stdErrPct": 5.7, "breakEvenPct": 52.3, "beatsBreakEven": False,
            "pricedRoiPct": 10.5,
        }}}
        prediction = self._prediction(ml_ev=4.0, total_ev=9.0)
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value=report):
            best = mlb_predictions.select_best_bet(prediction)
        total = next(o for o in best["options"] if o["market"] == "total")
        self.assertEqual(total["record"]["stdErrPct"], 5.7)
        self.assertFalse(total["record"]["beatsBreakEven"])

    def test_the_moneyline_carries_no_side_market_record(self) -> None:
        prediction = self._prediction(ml_ev=4.0)
        best = mlb_predictions.select_best_bet(prediction)
        self.assertIsNone(best["options"][0]["record"])

    def test_a_missing_record_gates_everything_off(self) -> None:
        """Absent evidence is not evidence of safety."""
        with patch.object(mlb_predictions, "_get_accuracy_report", return_value={}):
            self.assertFalse(mlb_predictions.market_is_validated("total"))
            self.assertFalse(mlb_predictions.market_is_validated("spread"))

    def test_an_unpriced_market_is_not_ranked_at_all(self) -> None:
        """No price means no expected value, so there is nothing to compare."""
        prediction = self._prediction(ml_ev=4.0)
        prediction["total"] = {"pick": "Over 8.5", "confidence": 58.0, "odds": None}
        with self._validated(total=True):
            best = mlb_predictions.select_best_bet(prediction)
        self.assertEqual([o["market"] for o in best["options"]], ["moneyline"])


class BoardRanksOnBestMarketTests(unittest.TestCase):
    """"Best play on the board" has to mean the best available bet.

    build_overview ranked every game by its moneyline EV, so a game whose total
    was the better wager was ranked by a number that was not the bet on offer --
    and the card then showed that moneyline EV beside a total nobody had priced.
    """

    def _game(self, event_id, matchup, ml_ev, total_ev):
        return {
            "eventId": event_id, "matchup": matchup, "homeTeam": "H", "awayTeam": "A",
            "prediction": {
                "predictedWinner": "H", "predictedSide": "home", "confidence": 70.0,
                "outcomeLabel": "H to win", "published": True,
                "value": {"evPct": ml_ev, "odds": -150, "kellyPct": 1.0, "breakEvenPct": 60.0},
                "total": {"pick": "Over 8.5", "confidence": 58.0, "odds": -108,
                          "value": {"evPct": total_ev, "odds": -108, "kellyPct": 2.0,
                                    "breakEvenPct": 51.9}},
            },
        }

    def _overview(self, games):
        from scripts.build_pages_data import build_overview

        with patch.object(mlb_predictions, "market_is_validated", side_effect=lambda m: True):
            for game in games:
                game["prediction"]["bestBet"] = mlb_predictions.select_best_bet(game["prediction"])
        return build_overview({"mlb": {"leagueLabel": "MLB", "games": games, "gameCount": len(games)}})

    def test_a_strong_total_outranks_a_stronger_moneyline_elsewhere(self) -> None:
        overview = self._overview([
            self._game("a", "Weak ML strong total", 2.0, 11.0),
            self._game("b", "Strong ML weak total", 7.0, 1.0),
        ])
        order = [(p["matchup"], p["betMarket"]) for p in overview["worthBacking"]]
        self.assertEqual(order[0], ("Weak ML strong total", "total"))
        self.assertEqual(order[1], ("Strong ML weak total", "moneyline"))

    def test_the_ranked_number_is_the_bet_on_offer(self) -> None:
        """Ranking by one number and displaying another is the original bug."""
        play = self._overview([self._game("a", "M", 2.0, 11.0)])["worthBacking"][0]
        self.assertAlmostEqual(play["evPct"], 11.0)
        self.assertEqual(play["odds"], -108, "the price shown must be the total's")
        self.assertEqual(play["betLabel"], "Over 8.5")

    def test_the_moneyline_number_is_still_available(self) -> None:
        """Losing it would make the model-vs-market rail unreadable."""
        play = self._overview([self._game("a", "M", 2.0, 11.0)])["worthBacking"][0]
        self.assertAlmostEqual(play["moneylineEvPct"], 2.0)
        self.assertEqual(play["pick"], "H", "the matchup rail is still a moneyline comparison")

    def test_a_moneyline_headline_leaves_betmarket_consistent(self) -> None:
        play = self._overview([self._game("a", "M", 7.0, 1.0)])["worthBacking"][0]
        self.assertEqual(play["betMarket"], "moneyline")
        self.assertAlmostEqual(play["evPct"], play["moneylineEvPct"])

    def test_a_game_with_no_priced_market_is_still_unpriced(self) -> None:
        game = self._game("a", "M", 2.0, 11.0)
        game["prediction"]["value"] = {}
        game["prediction"]["total"]["odds"] = None
        game["prediction"]["total"].pop("value")
        overview = self._overview([game])
        self.assertEqual(len(overview["unpriced"]), 1)
        self.assertIsNone(overview["unpriced"][0]["evPct"])


class PriceCoverageTests(unittest.TestCase):
    """"No prices" has two causes that look identical on the board.

    A feed that failed today comes back; a league no feed has ever covered does
    not. AFL is the second kind -- no SportsBookReview board and nothing in
    ESPN's core odds -- so all 56 of its graded picks are unpriced, permanently.
    Until now the only record of that was a stdout line that scrolls away in
    Actions and a comment in espn_odds.py, which makes it a claim rather than a
    measurement. Recording it per build means it is checked every run, and
    flips on its own if coverage ever appears.
    """

    def _coverage(self, league, **stats):
        from mlb_data import _price_source_coverage
        from sports_config import get_league

        return _price_source_coverage(get_league(league), stats)

    def test_a_league_with_no_feed_at_all_is_identified(self) -> None:
        coverage = self._coverage("afl", considered=10, priced=0)
        self.assertTrue(coverage["noSourceFound"])
        self.assertFalse(coverage["sbrBoard"])

    def test_it_flips_the_moment_a_price_appears(self) -> None:
        """The claim has to be falsifiable by the next build, not permanent."""
        self.assertFalse(self._coverage("afl", considered=10, priced=1)["noSourceFound"])

    def test_a_league_with_an_sbr_board_is_never_sourceless(self) -> None:
        """MLB going unpriced for a day is an outage, not an absent feed."""
        self.assertFalse(self._coverage("mlb", considered=40, priced=0)["noSourceFound"])

    def test_a_league_the_core_source_rescued_is_not_sourceless(self) -> None:
        coverage = self._coverage("wnba", considered=8, priced=6)
        self.assertFalse(coverage["noSourceFound"])
        self.assertEqual(coverage["espnCorePriced"], 6)

    def test_nothing_is_recorded_when_there_is_nothing_to_say(self) -> None:
        """A league with no board that was never even asked has established
        nothing, so it must not claim a verdict either way."""
        self.assertIsNone(self._coverage("afl"))

    def test_the_board_distinguishes_the_two_cases(self) -> None:
        from pathlib import Path

        js = (Path(__file__).resolve().parent.parent / "dashboard" / "board.js").read_text()
        self.assertIn("priceCoverage?.noSourceFound", js)
        self.assertIn("No odds feed covers this league", js)


class MlbSideMarketPricingTests(unittest.TestCase):
    """MLB side markets could never be priced, by construction.

    SportsBookReview posts MLB totals and spreads as bare numbers with no odds,
    and fill_missing_moneylines -- the only caller of ESPN core, which is the
    one source carrying side-market prices -- skips any game that already has a
    moneyline. SBR gives MLB a moneyline, so ESPN core was never asked about
    those games at all. The result was 11 priced spreads out of 58, the rest
    being baseball runlines with no price to value them at.
    """

    CORE = [
        {"sportsbook": "ESPN BET", "viewType": "MoneyLine",
         "currentLine": {"home": "-150", "away": "+130"}},
        {"sportsbook": "ESPN BET", "viewType": "Total",
         "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-112)"}},
        {"sportsbook": "ESPN BET", "viewType": "Spread",
         "currentLine": {"home": "-1.5 (+105)", "away": "+1.5 (-125)"}},
    ]

    def _sbr_game(self):
        """An MLB game exactly as SBR delivers it: moneyline priced, sides bare."""
        return {"eventId": "1", "lines": [
            {"sportsbook": "SBR", "viewType": "MoneyLine",
             "currentLine": {"home": "-150", "away": "+130"}},
            {"sportsbook": "SBR", "viewType": "Total", "currentLine": {"over": "8.5", "under": "8.5"}},
            {"sportsbook": "SBR", "viewType": "Spread", "currentLine": {"home": "-1.5", "away": "+1.5"}},
        ]}

    def setUp(self) -> None:
        import espn_odds

        espn_odds.clear_side_market_cache()

    def test_a_bare_sbr_line_is_recognised_as_unpriced(self) -> None:
        """A game can look fully covered and still be unvaluable."""
        import espn_odds

        self.assertFalse(espn_odds.has_priced_side_market(self._sbr_game()["lines"]))
        self.assertTrue(espn_odds.has_priced_side_market(self.CORE))

    def test_the_second_pass_prices_them(self) -> None:
        import espn_odds
        from mlb_predictions import extract_spread_price, extract_total_price

        game = self._sbr_game()
        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE):
            stats = espn_odds.fill_missing_side_market_prices([game], league="mlb")
        self.assertEqual(stats["priced"], 1)
        self.assertEqual(extract_total_price(game["lines"], "over"), -110)
        self.assertEqual(extract_spread_price(game["lines"], "home"), 105)

    def test_the_moneyline_is_not_duplicated(self) -> None:
        """A second copy would land in the de-vigged consensus twice."""
        import espn_odds

        game = self._sbr_game()
        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE):
            espn_odds.fill_missing_side_market_prices([game], league="mlb")
        moneylines = [l for l in game["lines"] if "MoneyLine" in l["viewType"]]
        self.assertEqual(len(moneylines), 1)

    def test_a_game_with_no_moneyline_is_left_to_the_other_pass(self) -> None:
        """fill_missing_moneylines owns those, and would fetch the same event."""
        import espn_odds

        game = {"eventId": "1", "lines": []}
        with patch.object(espn_odds, "fetch_event_odds", side_effect=AssertionError("must not fetch")):
            stats = espn_odds.fill_missing_side_market_prices([game], league="mlb")
        self.assertEqual(stats["considered"], 0)

    def test_an_already_priced_game_is_not_refetched(self) -> None:
        import espn_odds

        game = {"eventId": "1", "lines": self.CORE}
        with patch.object(espn_odds, "fetch_event_odds", side_effect=AssertionError("must not fetch")):
            stats = espn_odds.fill_missing_side_market_prices([game], league="mlb")
        self.assertEqual(stats["considered"], 0)

    def test_repeat_builds_reuse_the_cache(self) -> None:
        """MLB runs ~15 games a day against a build every thirty minutes;
        without a cache that is roughly 720 requests a day for lines that
        barely move."""
        import espn_odds

        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE) as fetch:
            for _ in range(5):
                espn_odds.fill_missing_side_market_prices([self._sbr_game()], league="mlb")
        self.assertEqual(fetch.call_count, 1)

    def test_the_cache_expires(self) -> None:
        import espn_odds

        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE) as fetch:
            espn_odds.fill_missing_side_market_prices([self._sbr_game()], league="mlb", now=0.0)
            espn_odds.fill_missing_side_market_prices(
                [self._sbr_game()], league="mlb",
                now=espn_odds.SIDE_MARKET_CACHE_TTL_SECONDS + 1,
            )
        self.assertEqual(fetch.call_count, 2)

    def test_it_reaches_a_real_runline_expected_value(self) -> None:
        """The whole point: an MLB runline had no price to be valued at."""
        import espn_odds

        game = self._sbr_game()
        game.update({"league": "mlb", "homeTeam": "Home", "awayTeam": "Away",
                     "homeRecord": "60-40", "awayRecord": "40-60", "enrichment": {}})
        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE):
            espn_odds.fill_missing_side_market_prices([game], league="mlb")
        mlb_predictions.apply_predictions([game])
        spread = game["prediction"].get("spread") or {}
        self.assertIsNotNone(spread.get("odds"), "the runline must now carry a price")
        self.assertIsNotNone((spread.get("value") or {}).get("evPct"))


class PerMarketPriceGuardTests(unittest.TestCase):
    """The guard has to be per market, not "any side market".

    The first version asked whether ANY side market carried a price. That is
    true for almost every MLB game, because the ESPN summary already prices
    totals -- so the fetch was skipped and the spread, the market that actually
    needed it, was never asked for. It shipped looking correct and produced
    nought of eighty-four priced MLB runlines on the live board, with the guard
    reporting everything fine.
    """

    CORE = [
        {"sportsbook": "ESPN BET", "viewType": "Total",
         "currentLine": {"over": "o8.5 (-110)", "under": "u8.5 (-112)"}},
        {"sportsbook": "ESPN BET", "viewType": "Spread",
         "currentLine": {"home": "-1.5 (+105)", "away": "+1.5 (-125)"}},
    ]

    def _live_mlb_shape(self):
        """Exactly what the live board had: total priced, spread bare."""
        return {"eventId": "1", "lines": [
            {"viewType": "MoneyLine", "currentLine": {"home": "-150", "away": "+130"}},
            {"viewType": "Total", "currentLine": {"over": "o8.5 (-108)", "under": "u8.5 (-112)"}},
            {"viewType": "Spread", "currentLine": {"home": "-1.5", "away": "+1.5"}},
        ]}

    def setUp(self) -> None:
        import espn_odds

        espn_odds.clear_side_market_cache()

    def test_a_priced_total_does_not_mask_an_unpriced_spread(self) -> None:
        import espn_odds

        lines = self._live_mlb_shape()["lines"]
        self.assertTrue(espn_odds.has_priced_market(lines, "Total"))
        self.assertFalse(espn_odds.has_priced_market(lines, "Spread"))

    def test_the_spread_is_fetched_even_when_the_total_is_priced(self) -> None:
        """The exact live failure: 0 of 84 MLB runlines priced."""
        import espn_odds
        from mlb_predictions import extract_spread_price

        game = self._live_mlb_shape()
        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE):
            stats = espn_odds.fill_missing_side_market_prices([game], league="mlb")
        self.assertEqual(stats["priced"], 1)
        self.assertEqual(extract_spread_price(game["lines"], "home"), 105)

    def test_an_already_priced_market_is_not_duplicated(self) -> None:
        """Re-adding a priced market puts the same line in the consensus twice."""
        import espn_odds

        game = self._live_mlb_shape()
        with patch.object(espn_odds, "fetch_event_odds", return_value=self.CORE):
            espn_odds.fill_missing_side_market_prices([game], league="mlb")
        totals = [l for l in game["lines"] if "Total" in l["viewType"]]
        self.assertEqual(len(totals), 1)

    def test_a_game_with_both_markets_priced_is_skipped(self) -> None:
        import espn_odds

        game = {"eventId": "1", "lines": [
            {"viewType": "MoneyLine", "currentLine": {"home": "-150", "away": "+130"}},
        ] + self.CORE}
        with patch.object(espn_odds, "fetch_event_odds", side_effect=AssertionError("must not fetch")):
            stats = espn_odds.fill_missing_side_market_prices([game], league="mlb")
        self.assertEqual(stats["considered"], 0)
