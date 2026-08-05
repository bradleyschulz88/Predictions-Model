"""Grading for totals and spreads -- markets that were never scored."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import accuracy_tracker  # noqa: E402
from accuracy_tracker import grade_spread, grade_total  # noqa: E402


class GradeTotalTests(unittest.TestCase):
    def _pick(self, side: str, line: float = 8.5) -> dict:
        return {"line": line, "pickSide": side}

    def test_over_wins_when_the_total_clears(self) -> None:
        self.assertEqual(grade_total(self._pick("over"), 6, 4)["outcome"], "win")

    def test_over_loses_when_it_does_not(self) -> None:
        self.assertEqual(grade_total(self._pick("over"), 3, 2)["outcome"], "loss")

    def test_under_is_the_mirror(self) -> None:
        self.assertEqual(grade_total(self._pick("under"), 3, 2)["outcome"], "win")
        self.assertEqual(grade_total(self._pick("under"), 6, 4)["outcome"], "loss")

    def test_landing_on_the_line_is_a_push(self) -> None:
        """A returned stake is neither a win nor a loss."""
        for side in ("over", "under"):
            self.assertEqual(grade_total(self._pick(side, 9), 5, 4)["outcome"], "push")

    def test_records_the_actual_total(self) -> None:
        self.assertEqual(grade_total(self._pick("over"), 6, 4)["actual"], 10)

    def test_missing_or_malformed_input_is_not_graded(self) -> None:
        self.assertIsNone(grade_total(None, 5, 4))
        self.assertIsNone(grade_total(self._pick("over"), None, 4))
        self.assertIsNone(grade_total({"line": 8.5}, 5, 4))
        self.assertIsNone(grade_total({"line": 8.5, "pickSide": "sideways"}, 5, 4))

    def test_a_priced_win_computes_units(self) -> None:
        pick = {"line": 8.5, "pickSide": "over", "odds": -110}
        result = grade_total(pick, 6, 4)
        self.assertEqual(result["odds"], -110)
        self.assertAlmostEqual(result["units"], 0.909, places=3)

    def test_a_priced_loss_computes_units(self) -> None:
        pick = {"line": 8.5, "pickSide": "over", "odds": -110}
        result = grade_total(pick, 3, 2)
        self.assertEqual(result["units"], -1.0)

    def test_a_priced_push_has_no_units(self) -> None:
        pick = {"line": 9, "pickSide": "over", "odds": -110}
        result = grade_total(pick, 5, 4)
        self.assertEqual(result["outcome"], "push")
        self.assertIsNone(result["units"])

    def test_an_unpriced_pick_has_no_units(self) -> None:
        result = grade_total(self._pick("over"), 6, 4)
        self.assertIsNone(result["odds"])
        self.assertIsNone(result["units"])


class GradeSpreadTests(unittest.TestCase):
    """`line` is the home number, so a home favourite carries a negative one."""

    def test_home_favourite_covers_by_winning_big_enough(self) -> None:
        pick = {"line": -1.5, "pickSide": "home"}
        self.assertEqual(grade_spread(pick, 5, 2)["outcome"], "win")

    def test_home_favourite_fails_to_cover_a_one_run_win(self) -> None:
        """The whole point of a runline: winning is not covering."""
        pick = {"line": -1.5, "pickSide": "home"}
        self.assertEqual(grade_spread(pick, 3, 2)["outcome"], "loss")

    def test_away_underdog_cashes_on_a_one_run_loss(self) -> None:
        pick = {"line": -1.5, "pickSide": "away"}
        self.assertEqual(grade_spread(pick, 3, 2)["outcome"], "win")

    def test_away_underdog_cashes_on_an_outright_win(self) -> None:
        pick = {"line": -1.5, "pickSide": "away"}
        self.assertEqual(grade_spread(pick, 2, 5)["outcome"], "win")

    def test_whole_number_line_can_push(self) -> None:
        """Impossible on baseball's -1.5, routine on a whole-number spread."""
        pick = {"line": -3.0, "pickSide": "home"}
        self.assertEqual(grade_spread(pick, 10, 7)["outcome"], "push")

    def test_a_no_lean_pick_is_not_a_position(self) -> None:
        self.assertIsNone(grade_spread({"line": -3.0, "pickSide": "push"}, 10, 7))

    def test_market_label_is_carried_through(self) -> None:
        pick = {"line": -1.5, "pickSide": "home", "market": "runline"}
        self.assertEqual(grade_spread(pick, 5, 2)["market"], "runline")

    def test_missing_input_is_not_graded(self) -> None:
        self.assertIsNone(grade_spread(None, 5, 2))
        self.assertIsNone(grade_spread({"line": -1.5, "pickSide": "home"}, None, 2))

    def test_a_priced_cover_computes_units(self) -> None:
        pick = {"line": -1.5, "pickSide": "home", "odds": +105}
        result = grade_spread(pick, 5, 2)
        self.assertEqual(result["odds"], 105)
        self.assertAlmostEqual(result["units"], 1.05, places=3)

    def test_an_unpriced_pick_has_no_units(self) -> None:
        pick = {"line": -1.5, "pickSide": "home"}
        result = grade_spread(pick, 5, 2)
        self.assertIsNone(result["odds"])
        self.assertIsNone(result["units"])


class EndToEndTests(unittest.TestCase):
    """Log a game with side markets, grade it, read the record back."""

    def _run(self, *, total_side: str, spread_side: str, home: int, away: int,
             total_line: float = 8.5, total_odds: int | None = None,
             spread_odds: int | None = None) -> dict:
        total_pick = {"line": total_line, "pickSide": total_side, "pick": "x", "detail": "ignored"}
        if total_odds is not None:
            total_pick["odds"] = total_odds
        spread_pick = {"line": -1.5, "pickSide": spread_side, "market": "runline"}
        if spread_odds is not None:
            spread_pick["odds"] = spread_odds
        payload = {
            "league": "mlb",
            "scheduleDate": "2026-07-28",
            "fetchedAt": "now",
            "games": [{
                "eventId": "42",
                "matchup": "Away @ Home",
                "prediction": {
                    "predictedWinner": "Home",
                    "predictedSide": "home",
                    "outcomeLabel": "Home to win",
                    "confidence": 70.0,
                    "features": {"league": "mlb"},
                    "total": total_pick,
                    "spread": spread_pick,
                },
            }],
        }
        game = {
            "eventId": "42", "homeTeam": "Home", "awayTeam": "Away",
            "homeScore": home, "awayScore": away,
            "isFinal": True, "isPostponed": False, "isCanceled": False,
            "isVoided": False, "isWashedOut": False, "isDelayed": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            accuracy_tracker.record_predictions(data_dir, [payload])
            with patch.object(accuracy_tracker, "fetch_scoreboard", return_value={}), \
                 patch.object(accuracy_tracker, "parse_scoreboard", return_value=[game]):
                return accuracy_tracker.grade_predictions(data_dir)

    def test_both_markets_are_graded_alongside_the_moneyline(self) -> None:
        accuracy = self._run(total_side="over", spread_side="home", home=6, away=4)
        record = accuracy["picksByEventId"]["42"]
        self.assertTrue(record["correct"], "moneyline")
        self.assertEqual(record["totalResult"]["outcome"], "win")
        self.assertEqual(record["spreadResult"]["outcome"], "win")

    def test_side_markets_can_lose_while_the_moneyline_wins(self) -> None:
        """The reason they are scored separately at all."""
        accuracy = self._run(total_side="over", spread_side="home", home=2, away=1)
        record = accuracy["picksByEventId"]["42"]
        self.assertTrue(record["correct"], "home won")
        self.assertEqual(record["totalResult"]["outcome"], "loss", "3 runs is under 8.5")
        self.assertEqual(record["spreadResult"]["outcome"], "loss", "won by 1, did not cover 1.5")

    def test_side_market_results_never_touch_the_moneyline_record(self) -> None:
        accuracy = self._run(total_side="over", spread_side="home", home=2, away=1)
        self.assertEqual(accuracy["summary"]["allTime"]["correct"], 1)
        self.assertEqual(accuracy["summary"]["allTime"]["total"], 1)

    def test_markets_are_summarised_separately(self) -> None:
        accuracy = self._run(total_side="over", spread_side="home", home=2, away=1)
        totals = accuracy["summary"]["totals"]
        spreads = accuracy["summary"]["spreads"]
        self.assertEqual(totals["graded"], 1)
        self.assertEqual(totals["losses"], 1)
        self.assertEqual(spreads["graded"], 1)
        self.assertEqual(spreads["losses"], 1)

    def test_summaries_claim_no_roi_when_nothing_is_priced(self) -> None:
        """No price is logged for these, so a return cannot be computed."""
        accuracy = self._run(total_side="over", spread_side="home", home=6, away=4)
        for key in ("totals", "spreads"):
            summary = accuracy["summary"][key]
            self.assertIsNone(summary["roiPct"])
            self.assertEqual(summary["priced"], 0)
            self.assertIn("not measurable", summary["note"])

    def test_a_priced_pick_carries_real_roi_through_the_summary(self) -> None:
        """ESPN core odds embedded a price for this one -- the summary should
        say so and compute a real return instead of falling back to hit rate."""
        accuracy = self._run(
            total_side="over", spread_side="home", home=6, away=4,
            total_odds=-110, spread_odds=-110,
        )
        for key in ("totals", "spreads"):
            summary = accuracy["summary"][key]
            self.assertEqual(summary["priced"], 1)
            self.assertEqual(summary["unpriced"], 0)
            self.assertAlmostEqual(summary["roiPct"], 90.9, delta=0.1)
            self.assertIn("both cover the full graded record", summary["note"])
        record = accuracy["picksByEventId"]["42"]
        self.assertEqual(record["totalResult"]["odds"], -110)
        self.assertEqual(record["spreadResult"]["odds"], -110)

    def test_pushes_are_excluded_from_the_hit_rate(self) -> None:
        """Counting a push as half a win lifts a break-even record above it.

        Needs a whole-number line -- a half-point total can never be pushed,
        which is exactly why books post them.
        """
        accuracy = self._run(
            total_side="over", spread_side="home", home=5, away=4, total_line=9
        )
        totals = accuracy["summary"]["totals"]
        self.assertEqual(totals["pushes"], 1)
        self.assertEqual(totals["graded"], 1)
        self.assertIsNone(totals["pct"], "a lone push leaves nothing decided")

    def test_log_keeps_only_what_grading_needs(self) -> None:
        """The log is committed every 30 minutes; reasoning text would bloat it."""
        payload_dir = tempfile.TemporaryDirectory()
        data_dir = Path(payload_dir.name)
        accuracy_tracker.record_predictions(data_dir, [{
            "league": "mlb", "scheduleDate": "2026-07-28", "fetchedAt": "now",
            "games": [{
                "eventId": "1", "matchup": "A @ B",
                "prediction": {
                    "predictedWinner": "B", "confidence": 70.0,
                    "total": {"line": 8.5, "pickSide": "over", "detail": "long text here"},
                },
            }],
        }])
        log = json.loads((data_dir / "predictions_log.json").read_text(encoding="utf-8"))
        stored = log["predictions"]["1"]["total"]
        self.assertEqual(stored, {"line": 8.5, "pickSide": "over"})
        payload_dir.cleanup()


class MarketSummaryTests(unittest.TestCase):
    """`_market_summary` directly, isolating the priced/unpriced convention
    from the rest of the grading pipeline."""

    def _row(self, key: str, *, outcome: str, odds=None, units=None) -> dict:
        row = {"outcome": outcome}
        if odds is not None:
            row["odds"] = odds
        if units is not None:
            row["units"] = units
        return {key: row}

    def test_all_unpriced_reports_no_measurable_roi(self) -> None:
        results = [
            self._row("totalResult", outcome="win"),
            self._row("totalResult", outcome="loss"),
        ]
        summary = accuracy_tracker._market_summary(results, "totalResult")
        self.assertEqual(summary["priced"], 0)
        self.assertIsNone(summary["roiPct"])
        self.assertEqual(summary["units"], 0.0)
        self.assertIn("not measurable", summary["note"])

    def test_all_priced_computes_roi_over_the_full_record(self) -> None:
        results = [
            self._row("totalResult", outcome="win", odds=-110, units=0.909),
            self._row("totalResult", outcome="loss", odds=-110, units=-1.0),
        ]
        summary = accuracy_tracker._market_summary(results, "totalResult")
        self.assertEqual(summary["priced"], 2)
        self.assertEqual(summary["unpriced"], 0)
        self.assertAlmostEqual(summary["units"], -0.091, places=3)
        self.assertAlmostEqual(summary["roiPct"], -4.55, delta=0.05)
        self.assertIn("both cover the full graded record", summary["note"])

    def test_mixed_priced_and_unpriced_counts_unpriced_as_zero_return(self) -> None:
        """The unpriced pick still counts in the denominator -- ROI reads as
        return per graded pick, not return per priced pick -- but contributes
        nothing to the numerator, mirroring the per-league moneyline summary."""
        results = [
            self._row("totalResult", outcome="win", odds=-110, units=0.909),
            self._row("totalResult", outcome="win"),  # unpriced
        ]
        summary = accuracy_tracker._market_summary(results, "totalResult")
        self.assertEqual(summary["graded"], 2)
        self.assertEqual(summary["priced"], 1)
        self.assertEqual(summary["unpriced"], 1)
        self.assertAlmostEqual(summary["units"], 0.909, places=3)
        self.assertAlmostEqual(summary["roiPct"], 45.45, delta=0.05)
        self.assertIn("1 of them", summary["note"])


class RetroactiveGradingTests(unittest.TestCase):
    """An already-graded record is never rebuilt, so it needs reconciling.

    Eight games sat graded with a total in the log and no result against it,
    because they graded before side-market scoring shipped. Without this the
    totals record would have started from whatever graded next and silently
    discarded every earlier pick -- the same failure the `published` flag had.
    """

    def _run(self, *, stored: dict, log_extras: dict) -> dict:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "accuracy.json").write_text(
                json.dumps({"picksByEventId": {"42": stored}}), encoding="utf-8"
            )
            entry = {
                "eventId": "42", "league": "mlb", "scheduleDate": "2026-07-26",
                "predictedWinner": "Home", "confidence": 70.0, "published": True,
            }
            entry.update(log_extras)
            (data_dir / "predictions_log.json").write_text(
                json.dumps({"predictions": {"42": entry}}), encoding="utf-8"
            )
            with patch.object(
                accuracy_tracker, "fetch_scoreboard", side_effect=RuntimeError("offline")
            ):
                return accuracy_tracker.grade_predictions(data_dir)

    def _graded(self, **extra) -> dict:
        row = {
            "eventId": "42", "league": "mlb", "scheduleDate": "2026-07-26",
            "predicted": "Home", "confidence": 70.0, "status": "graded",
            "correct": True, "homeScore": 11, "awayScore": 8, "gradedAt": "2026-07-27",
        }
        row.update(extra)
        return row

    def test_a_pre_existing_graded_row_gets_its_total_scored(self) -> None:
        accuracy = self._run(
            stored=self._graded(),
            log_extras={"total": {"line": 10.0, "pickSide": "over"}},
        )
        result = accuracy["picksByEventId"]["42"]["totalResult"]
        self.assertIsNotNone(result, "an old graded row must be reconciled")
        self.assertEqual(result["outcome"], "win", "19 runs clears 10.0")
        self.assertEqual(accuracy["summary"]["totals"]["wins"], 1)

    def test_a_pre_existing_graded_row_gets_its_spread_scored(self) -> None:
        accuracy = self._run(
            stored=self._graded(),
            log_extras={"spread": {"line": -1.5, "pickSide": "away", "market": "runline"}},
        )
        result = accuracy["picksByEventId"]["42"]["spreadResult"]
        self.assertEqual(result["outcome"], "loss", "home won by 3, so away +1.5 loses")

    def test_an_existing_result_is_not_recomputed(self) -> None:
        """Reconciliation must fill gaps, not overwrite settled history."""
        settled = {"line": 10.0, "pickSide": "over", "actual": 19, "outcome": "win"}
        accuracy = self._run(
            stored=self._graded(totalResult=settled),
            log_extras={"total": {"line": 99.0, "pickSide": "under"}},
        )
        self.assertEqual(accuracy["picksByEventId"]["42"]["totalResult"], settled)

    def test_a_row_without_scores_is_left_alone(self) -> None:
        accuracy = self._run(
            stored=self._graded(homeScore=None, awayScore=None),
            log_extras={"total": {"line": 10.0, "pickSide": "over"}},
        )
        self.assertIsNone(accuracy["picksByEventId"]["42"].get("totalResult"))

    def test_a_pending_row_is_not_graded_early(self) -> None:
        pending = self._graded(status="pending", correct=None)
        accuracy = self._run(
            stored=pending, log_extras={"total": {"line": 10.0, "pickSide": "over"}}
        )
        self.assertIsNone(accuracy["picksByEventId"]["42"].get("totalResult"))


class MarketUncertaintyTests(unittest.TestCase):
    """A hit rate on a few dozen picks is not a settled number.

    The spread record fell from 67.9% to 58.7% inside one afternoon's grading,
    which is exactly the regression a thin sample predicts. Reporting the rate
    with no interval, and an ROI diluted across picks that could never have
    contributed to it, presented that as fact.
    """

    def _rows(self, key, wins, losses, *, odds=-110, priced=None):
        """`priced` many rows carry a price; the rest carry none."""
        priced = wins + losses if priced is None else priced
        rows = []
        for index in range(wins + losses):
            outcome = "win" if index < wins else "loss"
            row = {"outcome": outcome}
            if index < priced:
                row["odds"] = odds
                row["units"] = accuracy_tracker.american_odds_profit(odds, outcome == "win")
            rows.append({key: row})
        return rows

    def test_standard_error_shrinks_as_the_sample_grows(self) -> None:
        small = accuracy_tracker._market_summary(self._rows("totalResult", 33, 21), "totalResult")
        large = accuracy_tracker._market_summary(self._rows("totalResult", 330, 210), "totalResult")
        self.assertAlmostEqual(small["pct"], large["pct"], places=0)
        self.assertGreater(small["stdErrPct"], large["stdErrPct"])
        self.assertAlmostEqual(small["stdErrPct"], large["stdErrPct"] * (10 ** 0.5), delta=0.2)

    def test_a_thin_winning_record_is_not_called_conclusive(self) -> None:
        """61% on 54 decided has a 95% interval that still spans break-even."""
        summary = accuracy_tracker._market_summary(self._rows("totalResult", 33, 21), "totalResult")
        self.assertGreater(summary["pct"], summary["breakEvenPct"])
        self.assertFalse(summary["beatsBreakEven"])

    def test_the_same_rate_on_a_big_sample_is_conclusive(self) -> None:
        summary = accuracy_tracker._market_summary(self._rows("totalResult", 330, 210), "totalResult")
        self.assertTrue(summary["beatsBreakEven"])

    def test_a_losing_record_is_never_conclusive(self) -> None:
        summary = accuracy_tracker._market_summary(self._rows("totalResult", 210, 330), "totalResult")
        self.assertFalse(summary["beatsBreakEven"])

    def test_break_even_comes_from_the_prices_actually_taken(self) -> None:
        """Hardcoding -110 is fine until MLB runlines carry prices, which sit
        nearer +150/-200 and imply a very different bar."""
        cheap = accuracy_tracker._market_summary(
            self._rows("spreadResult", 30, 30, odds=-110), "spreadResult")
        dear = accuracy_tracker._market_summary(
            self._rows("spreadResult", 30, 30, odds=-250), "spreadResult")
        self.assertAlmostEqual(cheap["breakEvenPct"], 52.4, delta=0.2)
        self.assertAlmostEqual(dear["breakEvenPct"], 71.4, delta=0.2)

    def test_break_even_falls_back_to_minus_110_when_nothing_is_priced(self) -> None:
        summary = accuracy_tracker._market_summary(
            self._rows("totalResult", 5, 5, priced=0), "totalResult")
        self.assertEqual(summary["breakEvenPct"], accuracy_tracker.DEFAULT_BREAK_EVEN_PCT)

    def test_priced_roi_is_reported_separately_from_the_diluted_one(self) -> None:
        """The headline divides units by every graded pick while only the
        priced ones can contribute, so a market with 10 prices out of 75
        reports a modest return that is really ten picks' worth of evidence."""
        rows = self._rows("spreadResult", 40, 35, priced=10)
        summary = accuracy_tracker._market_summary(rows, "spreadResult")
        self.assertEqual(summary["priced"], 10)
        self.assertNotAlmostEqual(summary["roiPct"], summary["pricedRoiPct"], places=1)
        self.assertAlmostEqual(
            summary["pricedRoiPct"], summary["pricedUnits"] / 10 * 100, places=1
        )
        self.assertIn("priced picks alone", summary["note"])

    def test_a_fully_priced_market_reports_the_same_roi_both_ways(self) -> None:
        rows = self._rows("totalResult", 30, 20)
        summary = accuracy_tracker._market_summary(rows, "totalResult")
        self.assertAlmostEqual(summary["roiPct"], summary["pricedRoiPct"], places=1)

    def test_an_empty_market_does_not_divide_by_zero(self) -> None:
        summary = accuracy_tracker._market_summary([], "totalResult")
        self.assertEqual(summary["graded"], 0)
        self.assertIsNone(summary["pct"])
        self.assertIsNone(summary["stdErrPct"])
        self.assertFalse(summary["beatsBreakEven"])


if __name__ == "__main__":
    unittest.main()
