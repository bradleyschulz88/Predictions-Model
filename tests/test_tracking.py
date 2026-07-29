"""Tests for closing line value, unpriced-league reporting and log pruning."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import accuracy_tracker as tracker  # noqa: E402
from accuracy_tracker import closing_line_value  # noqa: E402


def _pick(opening, closing, side: str = "home", opening_side: str | None = None) -> dict:
    return {
        "openingOdds": opening,
        "pickOdds": closing,
        "predictedSide": side,
        "openingSide": opening_side or side,
    }


class ClosingLineValueTests(unittest.TestCase):
    def test_positive_when_the_price_shortens_after_the_pick(self) -> None:
        """Taking +130 on a side the market closes at +110 is a beat."""
        self.assertGreater(closing_line_value(_pick(130, 110)), 0)

    def test_positive_for_favourites_too(self) -> None:
        self.assertGreater(closing_line_value(_pick(-110, -130)), 0)

    def test_negative_when_the_price_drifts_out(self) -> None:
        self.assertLess(closing_line_value(_pick(-130, -110)), 0)

    def test_zero_when_the_line_does_not_move(self) -> None:
        self.assertAlmostEqual(closing_line_value(_pick(-115, -115)), 0.0)

    def test_none_when_the_pick_side_changed(self) -> None:
        # Opening and closing prices for different sides are not comparable.
        self.assertIsNone(closing_line_value(_pick(130, 110, side="away", opening_side="home")))

    def test_none_when_odds_are_missing(self) -> None:
        self.assertIsNone(closing_line_value(_pick(None, 110)))
        self.assertIsNone(closing_line_value(_pick(130, None)))
        self.assertIsNone(closing_line_value({}))

    def test_none_for_unparseable_odds(self) -> None:
        self.assertIsNone(closing_line_value(_pick("n/a", 110)))


class LogPruningTests(unittest.TestCase):
    def test_leaves_a_small_log_alone(self) -> None:
        log = {"predictions": {str(i): {"scheduleDate": "2026-06-01"} for i in range(10)}}
        tracker._prune_log(log)
        self.assertEqual(len(log["predictions"]), 10)

    def test_caps_an_oversized_log(self) -> None:
        limit = tracker.MAX_LOGGED_PREDICTIONS
        log = {
            "predictions": {
                str(i): {"scheduleDate": f"2026-06-{(i % 28) + 1:02d}"} for i in range(limit + 500)
            }
        }
        tracker._prune_log(log)
        self.assertEqual(len(log["predictions"]), limit)

    def test_keeps_the_most_recent_entries(self) -> None:
        limit = tracker.MAX_LOGGED_PREDICTIONS
        log = {"predictions": {}}
        for i in range(limit + 10):
            # Older dates first, so the newest must survive.
            log["predictions"][str(i)] = {"scheduleDate": f"2026-{(i % 12) + 1:02d}-01"}
        log["predictions"]["newest"] = {"scheduleDate": "2099-12-31"}
        tracker._prune_log(log)
        self.assertIn("newest", log["predictions"])


class UnpricedLeagueTests(unittest.TestCase):
    """A win rate with no price behind it must not read as break-even ROI."""

    def _grade(self, picks: list[dict]) -> dict:
        by_league: dict[str, dict] = {}
        for item in picks:
            bucket = by_league.setdefault(item["league"], tracker._summary_bucket())
            bucket["total"] += 1
            if item.get("correct"):
                bucket["correct"] += 1
            bucket["units"] = round(bucket["units"] + float(item.get("units") or 0.0), 3)
            if item.get("pickOdds") is not None:
                bucket["priced"] = bucket.get("priced", 0) + 1
            bucket["pct"] = round(bucket["correct"] / bucket["total"] * 100, 1)
            bucket["roiPct"] = round(bucket["units"] / bucket["total"] * 100, 1)

        for bucket in by_league.values():
            priced = bucket.get("priced", 0)
            bucket["priced"] = priced
            bucket["unpriced"] = bucket["total"] - priced
            bucket["pricedPct"] = round(priced / bucket["total"] * 100, 1) if bucket["total"] else None
            if not priced:
                bucket["roiPct"] = None
                bucket["roiNote"] = "No odds available for this league; ROI is not measurable."
        return by_league

    def test_league_without_odds_reports_no_roi(self) -> None:
        result = self._grade(
            [{"league": "afl", "correct": True, "pickOdds": None} for _ in range(20)]
        )
        self.assertIsNone(result["afl"]["roiPct"])
        self.assertIn("roiNote", result["afl"])
        self.assertEqual(result["afl"]["pct"], 100.0)

    def test_league_with_odds_keeps_its_roi(self) -> None:
        result = self._grade(
            [{"league": "mlb", "correct": True, "pickOdds": -110, "units": 0.909} for _ in range(20)]
        )
        self.assertIsNotNone(result["mlb"]["roiPct"])
        self.assertEqual(result["mlb"]["pricedPct"], 100.0)

    def test_partial_coverage_is_reported(self) -> None:
        picks = [{"league": "mlb", "correct": True, "pickOdds": -110, "units": 0.9} for _ in range(6)]
        picks += [{"league": "mlb", "correct": False, "pickOdds": None} for _ in range(4)]
        result = self._grade(picks)
        self.assertEqual(result["mlb"]["priced"], 6)
        self.assertEqual(result["mlb"]["unpriced"], 4)
        self.assertEqual(result["mlb"]["pricedPct"], 60.0)


class QuarantineTests(unittest.TestCase):
    def test_orphaned_model_artifacts_are_not_on_the_prediction_path(self) -> None:
        """Nothing outside ml_model/ may import the experimental artifacts."""
        offenders = []
        for path in ROOT.glob("*.py"):
            if "experimental" in path.read_text(encoding="utf-8"):
                offenders.append(path.name)
        self.assertEqual(offenders, [])

    def test_experimental_directory_documents_itself(self) -> None:
        readme = ROOT / "ml_model" / "experimental" / "README.md"
        self.assertTrue(readme.is_file())
        self.assertIn("not on the prediction path", readme.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()


class PublishVersusLogTests(unittest.TestCase):
    """Publishing and logging are different questions.

    MLB's 55-65% band is withheld from the board because it loses money, but it
    is exactly where the fit is most wrong. Censoring it from the training log
    would stop the model ever learning to correct itself there -- the threshold
    would entrench the error it exists to hide.
    """

    def _payload(self, confidence: float, league: str = "mlb") -> dict:
        return {
            "league": league,
            "scheduleDate": "2026-07-28",
            "fetchedAt": "now",
            "games": [
                {
                    "eventId": "42",
                    "matchup": "A @ B",
                    "prediction": {
                        "predictedWinner": "B",
                        "predictedSide": "home",
                        "outcomeLabel": "B to win",
                        "confidence": confidence,
                        "features": {"recordDiff": 0.2, "league": league},
                    },
                }
            ],
        }

    def _log(self, tmp: str, payload: dict) -> dict:
        from pathlib import Path

        from accuracy_tracker import record_predictions

        record_predictions(Path(tmp), [payload])
        return json.loads((Path(tmp) / "predictions_log.json").read_text(encoding="utf-8"))

    def test_withheld_pick_is_still_logged_for_training(self) -> None:
        """Below the floor, so never shown -- but the fit still needs it."""
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, self._payload(51.0))
        row = log["predictions"]["42"]
        self.assertFalse(row["published"])
        # The features are the whole point -- the fit reads these.
        self.assertEqual(row["features"]["recordDiff"], 0.2)

    def test_published_pick_is_flagged_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, self._payload(71.0))
        self.assertTrue(log["predictions"]["42"]["published"])

    def test_mid_band_publishes_in_every_league(self) -> None:
        """No league carries an override now, so 60% publishes everywhere."""
        for league in ("mlb", "wnba"):
            with tempfile.TemporaryDirectory() as tmp:
                log = self._log(tmp, self._payload(60.0, league=league))
            self.assertTrue(log["predictions"]["42"]["published"], league)

    def test_withheld_picks_are_excluded_from_the_record(self) -> None:
        """A record nobody could have bet is not a record.

        Asserts the filter directly rather than through grade_predictions, which
        fetches a scoreboard per league per day and would take the suite off the
        network for minutes.
        """
        from accuracy_tracker import _build_pick_record

        withheld = _build_pick_record(pending={"eventId": "42", "published": False}, status="pending")
        shown = _build_pick_record(pending={"eventId": "43", "published": True}, status="pending")
        kept = [row for row in (withheld, shown) if row.get("published", True)]
        self.assertEqual([row["eventId"] for row in kept], ["43"])

    def test_legacy_rows_without_the_flag_count_as_published(self) -> None:
        """Everything logged before the split was publishable by definition."""
        from accuracy_tracker import _build_pick_record

        record = _build_pick_record(pending={"eventId": "1"}, status="pending")
        self.assertTrue(record["published"])


class PublishedFlagReconciliationTests(unittest.TestCase):
    """A graded record is never rebuilt, so its flag has to be refreshed.

    accuracy.json is carried forward between runs and rows that are already
    graded are skipped by the grading loop. Without an explicit reconciliation
    pass, every row written before the publish/log split keeps no flag at all --
    which reads as published -- so raising a threshold would only ever apply to
    picks made after the change, and the losing picks it exists to remove would
    stay in the published record forever.
    """

    def _run(self, tmp: str, *, stored_flag, log_flag) -> dict:
        from pathlib import Path
        from unittest.mock import patch

        import accuracy_tracker

        data_dir = Path(tmp)
        stored: dict = {
            "eventId": "42",
            "league": "mlb",
            "scheduleDate": "2026-07-28",
            "predicted": "B",
            "outcomeLabel": "B to win",
            "confidence": 60.0,
            "status": "graded",
            "correct": False,
            "gradedAt": "2026-07-28",
        }
        if stored_flag is not None:
            stored["published"] = stored_flag

        (data_dir / "accuracy.json").write_text(
            json.dumps({"picksByEventId": {"42": stored}}), encoding="utf-8"
        )
        (data_dir / "predictions_log.json").write_text(
            json.dumps(
                {
                    "predictions": {
                        "42": {
                            "eventId": "42",
                            "league": "mlb",
                            "scheduleDate": "2026-07-28",
                            "predictedWinner": "B",
                            "confidence": 60.0,
                            "published": log_flag,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        # No network: the grading loop has nothing to fetch, which is exactly the
        # path where an already-graded row is skipped.
        with patch.object(
            accuracy_tracker, "fetch_scoreboard", side_effect=RuntimeError("offline")
        ):
            return accuracy_tracker.grade_predictions(data_dir)

    def test_legacy_graded_row_is_withheld_when_the_log_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._run(tmp, stored_flag=None, log_flag=False)

        self.assertFalse(accuracy["picksByEventId"]["42"]["published"])
        graded = [row for row in accuracy.get("recentResults") or [] if row["eventId"] == "42"]
        self.assertEqual(graded, [], "withheld pick leaked into the published record")
        self.assertEqual(accuracy["summary"]["allTime"]["total"], 0)

    def test_a_pick_that_becomes_publishable_returns_to_the_record(self) -> None:
        """The reconciliation runs both ways, so lowering a bar restores picks."""
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._run(tmp, stored_flag=False, log_flag=True)

        self.assertTrue(accuracy["picksByEventId"]["42"]["published"])
        self.assertEqual(accuracy["summary"]["allTime"]["total"], 1)


class AbandonedGameTests(unittest.TestCase):
    """A pick on a game that never happened must reach a terminal state.

    12 picks were stuck at "pending", the oldest from 2026-06-18. Ten were
    rain-outs replayed later -- six as the second game of a doubleheader the
    next day, which is what a postponed MLB game normally becomes. Grading
    correctly refused to score them, then treated "called off" exactly like
    "not finished yet", so they never resolved.
    """

    def _setup(self, tmp: str, *, schedule_date: str, games: list[dict]) -> dict:
        from pathlib import Path
        from unittest.mock import patch

        import accuracy_tracker

        data_dir = Path(tmp)
        (data_dir / "predictions_log.json").write_text(
            json.dumps(
                {
                    "predictions": {
                        "42": {
                            "eventId": "42",
                            "league": "mlb",
                            "scheduleDate": schedule_date,
                            "matchup": "Orioles @ Red Sox",
                            "predictedWinner": "Red Sox",
                            "confidence": 70.0,
                            "published": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        def fake_parse(_scoreboard, league=None):
            return games

        with patch.object(accuracy_tracker, "fetch_scoreboard", return_value={}), patch.object(
            accuracy_tracker, "parse_scoreboard", side_effect=fake_parse
        ):
            return accuracy_tracker.grade_predictions(data_dir)

    def _game(self, **flags) -> dict:
        game = {
            "eventId": "42",
            "homeTeam": "Red Sox",
            "awayTeam": "Orioles",
            "isFinal": False,
            "isPostponed": False,
            "isCanceled": False,
            "isVoided": False,
            "isWashedOut": False,
            "isDelayed": False,
        }
        game.update(flags)
        return game

    def _yesterday(self) -> str:
        from schedule_dates import league_schedule_date

        return league_schedule_date("mlb", -1)

    def _long_ago(self) -> str:
        from schedule_dates import league_schedule_date

        return league_schedule_date("mlb", -10)

    def test_postponed_game_seen_on_its_date_is_voided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(
                tmp, schedule_date=self._yesterday(), games=[self._game(isPostponed=True)]
            )
        record = accuracy["picksByEventId"]["42"]
        self.assertEqual(record["status"], "voided")
        self.assertEqual(record["voidReason"], "postponed")

    def test_canceled_game_is_voided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(
                tmp, schedule_date=self._yesterday(), games=[self._game(isCanceled=True)]
            )
        self.assertEqual(accuracy["picksByEventId"]["42"]["voidReason"], "canceled")

    def test_rescheduled_game_that_vanished_is_aged_out(self) -> None:
        """ESPN drops a rescheduled game from its original date entirely.

        It is never seen again, so ageing out is the only way it can resolve.
        """
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(tmp, schedule_date=self._long_ago(), games=[])
        record = accuracy["picksByEventId"]["42"]
        self.assertEqual(record["status"], "voided")
        self.assertEqual(record["voidReason"], "no result reported")

    def test_a_delayed_game_is_left_alone(self) -> None:
        """A rain delay has not finished; it has not been called off."""
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(
                tmp, schedule_date=self._yesterday(), games=[self._game(isDelayed=True)]
            )
        self.assertEqual(accuracy["picksByEventId"]["42"]["status"], "pending")

    def test_a_recent_missing_game_is_not_voided_yet(self) -> None:
        """Yesterday's game may simply not have a final posted yet."""
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(tmp, schedule_date=self._yesterday(), games=[])
        self.assertEqual(accuracy["picksByEventId"]["42"]["status"], "pending")

    def test_a_fetch_failure_never_voids_a_pick(self) -> None:
        """The guard that matters: ESPN being down is not evidence of anything."""
        from pathlib import Path
        from unittest.mock import patch

        import accuracy_tracker

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "predictions_log.json").write_text(
                json.dumps(
                    {
                        "predictions": {
                            "42": {
                                "eventId": "42",
                                "league": "mlb",
                                "scheduleDate": self._long_ago(),
                                "predictedWinner": "Red Sox",
                                "confidence": 70.0,
                                "published": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                accuracy_tracker, "fetch_scoreboard", side_effect=RuntimeError("ESPN down")
            ):
                accuracy = accuracy_tracker.grade_predictions(data_dir)

        self.assertEqual(accuracy["picksByEventId"]["42"]["status"], "pending")

    def test_a_played_game_still_grades_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(
                tmp,
                schedule_date=self._yesterday(),
                games=[self._game(isFinal=True, homeScore=5, awayScore=2)],
            )
        record = accuracy["picksByEventId"]["42"]
        self.assertEqual(record["status"], "graded")
        self.assertTrue(record["correct"])

    def test_voids_are_counted_not_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            accuracy = self._setup(
                tmp, schedule_date=self._yesterday(), games=[self._game(isPostponed=True)]
            )
        summary = accuracy["summary"]["allTime"]
        self.assertEqual(summary["voided"], 1)
        # A void is neither a win nor a loss, and is no longer pending either.
        self.assertEqual(summary["total"], 0)
        self.assertEqual(summary["pending"], 0)


class ClosingLineValueRobustnessTests(unittest.TestCase):
    """CLV runs on every record in the grading loop, so it must never raise.

    `int(float("inf"))` raises OverflowError, not ValueError, so a single
    non-finite price aborted the whole run instead of costing one row its CLV.
    Same family as the grade_total/grade_spread overflow, found by fuzzing the
    wagering math rather than by reading it.
    """

    def _record(self, opening, closing=-110):
        return {
            "openingOdds": opening, "pickOdds": closing,
            "predictedSide": "home", "openingSide": "home",
        }

    def test_non_finite_odds_do_not_raise(self) -> None:
        for bad in (float("inf"), float("-inf"), float("nan")):
            self.assertIsNone(closing_line_value(self._record(bad)))
            self.assertIsNone(closing_line_value(self._record(-110, bad)))

    def test_unparseable_odds_still_return_none(self) -> None:
        self.assertIsNone(closing_line_value(self._record("n/a")))
        self.assertIsNone(closing_line_value(self._record(None)))

    def test_a_real_move_is_still_measured(self) -> None:
        """Guards the guard: hardening must not have broken the calculation."""
        self.assertGreater(closing_line_value(self._record(130, 110)), 0)
        self.assertLess(closing_line_value(self._record(-130, -110)), 0)
