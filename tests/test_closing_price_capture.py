"""The closing price must survive the build that runs after the game starts.

CLV is the metric that says whether this model makes money, and it was being
measured against a price that two ordinary events could destroy.

Nothing stopped a build from overwriting pickOdds once a game was under way, so
a build landing mid-game replaced the closing line with an in-play number and
one landing after the final replaced it with whatever the book showed then.
Separately, any build whose odds fetch came back empty wrote None straight over
a price already recorded. Neither failure was visible: both produce a plausible
number, or no number, with no warning.

The rule is now that a recorded price is never replaced by nothing, and never
updated once the game is under way.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import clv_summary, record_predictions  # noqa: E402


def _payload(odds, *, live=False, final=False, voided=False, fetched="2026-08-05T12:00:00Z"):
    lines = []
    if odds is not None:
        lines = [{
            "sportsbook": "TestBook",
            "viewType": "MoneyLine",
            "currentLine": {"homeOdds": odds, "awayOdds": 100},
        }]
    return {
        "league": "mlb",
        "scheduleDate": "2026-08-05",
        "fetchedAt": fetched,
        "games": [{
            "eventId": "999",
            "league": "mlb",
            "matchup": "Away @ Home",
            "homeTeam": "Home",
            "awayTeam": "Away",
            "isLive": live,
            "isFinal": final,
            "isVoided": voided,
            "lines": lines,
            "prediction": {
                "predictedWinner": "Home",
                "predictedSide": "home",
                "confidence": 60.0,
                "features": {},
            },
        }],
    }


class ClosingPriceCaptureTests(unittest.TestCase):
    def _run(self, *payloads) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for payload in payloads:
                record_predictions(data_dir, [payload])
            log = json.loads((data_dir / "predictions_log.json").read_text())
        return log["predictions"]["999"]

    def test_price_still_moves_while_the_game_is_upcoming(self) -> None:
        """The fix must not freeze the line early -- that is the whole signal."""
        entry = self._run(_payload(-120), _payload(-140))
        self.assertEqual(entry["openingOdds"], -120)
        self.assertEqual(entry["pickOdds"], -140)
        self.assertIsNone(entry["pickOddsFrozenAt"])

    def test_in_play_price_does_not_overwrite_the_close(self) -> None:
        entry = self._run(_payload(-120), _payload(-140), _payload(+250, live=True))
        self.assertEqual(entry["pickOdds"], -140)

    def test_post_final_price_does_not_overwrite_the_close(self) -> None:
        entry = self._run(_payload(-120), _payload(-140), _payload(-105, final=True))
        self.assertEqual(entry["pickOdds"], -140)

    def test_freeze_is_stamped_once_and_does_not_drift(self) -> None:
        entry = self._run(
            _payload(-140),
            _payload(-150, final=True, fetched="2026-08-05T22:00:00Z"),
            _payload(-160, final=True, fetched="2026-08-06T02:00:00Z"),
        )
        self.assertEqual(entry["pickOddsFrozenAt"], "2026-08-05T22:00:00Z")
        self.assertEqual(entry["pickOdds"], -140)

    def test_a_missing_price_never_erases_a_recorded_one(self) -> None:
        """A provider blip used to cost the price permanently."""
        entry = self._run(_payload(-120), _payload(None))
        self.assertEqual(entry["pickOdds"], -120)
        self.assertEqual(entry["openingOdds"], -120)

    def test_price_recovers_after_a_blip_while_still_upcoming(self) -> None:
        entry = self._run(_payload(-120), _payload(None), _payload(-135))
        self.assertEqual(entry["pickOdds"], -135)

    def test_a_game_first_seen_after_it_started_still_records_its_price(self) -> None:
        """Freezing must not mean never recording; there is nothing to protect."""
        entry = self._run(_payload(-130, final=True))
        self.assertEqual(entry["pickOdds"], -130)

    def test_voided_game_keeps_its_pre_game_price(self) -> None:
        entry = self._run(_payload(-120), _payload(-200, voided=True))
        self.assertEqual(entry["pickOdds"], -120)




class ClvReportingTests(unittest.TestCase):
    """A CLV number nobody can act on is worse than no number.

    The headline was -0.52% over 96 picks, mixing genuine closes with the
    latest quote seen. These call the production function rather than a copy
    of its arithmetic -- a test that reimplements the thing it checks passes
    against itself, which is how this kind of defect survives.
    """

    @staticmethod
    def _pick(clv, frozen):
        return {"clvPct": clv, "pickOddsFrozenAt": frozen}

    def test_unfrozen_picks_are_counted_separately_not_averaged_in(self) -> None:
        summary = clv_summary([
            self._pick(5.0, "2026-08-05T22:00:00Z"),
            self._pick(-90.0, None),
        ])
        self.assertEqual(summary["picks"], 1)
        self.assertEqual(summary["provisionalPicks"], 1)
        self.assertEqual(summary["avgPct"], 5.0)

    def test_no_confirmed_closes_reports_nothing_rather_than_zero(self) -> None:
        summary = clv_summary([self._pick(3.0, None), self._pick(-3.0, None)])
        self.assertEqual(summary["picks"], 0)
        self.assertIsNone(summary["avgPct"])
        self.assertIsNone(summary["beatCloseP"])
        self.assertIsNone(summary["beatsCoinFlip"])
        self.assertEqual(summary["provisionalPicks"], 2)

    def test_a_thin_sample_near_even_is_not_called_a_win(self) -> None:
        window = [self._pick(1.0, "t")] * 11 + [self._pick(-1.0, "t")] * 9
        summary = clv_summary(window)
        self.assertEqual(summary["beatCloseP"], 55.0)
        self.assertFalse(summary["beatsCoinFlip"])

    def test_a_strong_sample_does_clear_the_bar(self) -> None:
        window = [self._pick(1.0, "t")] * 290 + [self._pick(-1.0, "t")] * 210
        summary = clv_summary(window)
        self.assertEqual(summary["beatCloseP"], 58.0)
        self.assertTrue(summary["beatsCoinFlip"])

    def test_the_current_headline_would_not_have_cleared_the_bar(self) -> None:
        """46.9% on 96 picks was never evidence of anything."""
        window = [self._pick(1.0, "t")] * 45 + [self._pick(-1.0, "t")] * 51
        summary = clv_summary(window)
        self.assertFalse(summary["beatsCoinFlip"])


if __name__ == "__main__":
    unittest.main()
