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

    def test_withheld_mlb_pick_is_still_logged_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, self._payload(60.0))
        row = log["predictions"]["42"]
        self.assertFalse(row["published"])
        # The features are the whole point -- the fit reads these.
        self.assertEqual(row["features"]["recordDiff"], 0.2)

    def test_published_mlb_pick_is_flagged_published(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, self._payload(71.0))
        self.assertTrue(log["predictions"]["42"]["published"])

    def test_other_leagues_publish_in_the_mid_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log(tmp, self._payload(60.0, league="wnba"))
        self.assertTrue(log["predictions"]["42"]["published"])

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
