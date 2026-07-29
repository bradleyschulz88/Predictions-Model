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


class EndToEndTests(unittest.TestCase):
    """Log a game with side markets, grade it, read the record back."""

    def _run(self, *, total_side: str, spread_side: str, home: int, away: int,
             total_line: float = 8.5) -> dict:
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
                    "total": {"line": total_line, "pickSide": total_side, "pick": "x", "detail": "ignored"},
                    "spread": {"line": -1.5, "pickSide": spread_side, "market": "runline"},
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

    def test_summaries_claim_no_roi(self) -> None:
        """No price is logged for these, so a return cannot be computed."""
        accuracy = self._run(total_side="over", spread_side="home", home=6, away=4)
        for key in ("totals", "spreads"):
            self.assertNotIn("roiPct", accuracy["summary"][key])
            self.assertIn("not measurable", accuracy["summary"][key]["note"])

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


if __name__ == "__main__":
    unittest.main()
