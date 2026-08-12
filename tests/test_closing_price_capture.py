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


class NegativeClvIsReportableTests(unittest.TestCase):
    """A significantly bad reading must not render as "no evidence".

    `beatsCoinFlip` alone answers False for two completely different states:
    "provably worse than chance" and "too thin to say". So a rate two standard
    errors on the wrong side of 50 looked exactly like an inconclusive one --
    which is how the model sat at 38.9% over 90 confirmed picks, an interval of
    28.5-49.3 that excludes 50, while the board reported +2.6% return and the
    CLV card said "not yet distinguishable from a coin flip".

    Measured 12 Aug 2026 on the live record: 90 confirmed closes, 35 beat, 52
    lost, 3 unmoved; median -0.61% against a mean of -0.16%; MLB 35.4% on 65.
    """

    @staticmethod
    def _pick(clv, frozen="t", league="mlb"):
        return {"clvPct": clv, "pickOddsFrozenAt": frozen, "league": league}

    def _record(self, beat, lost, unmoved=0, league="mlb"):
        return ([self._pick(1.0, league=league)] * beat
                + [self._pick(-1.0, league=league)] * lost
                + [self._pick(0.0, league=league)] * unmoved)

    def test_a_significantly_bad_rate_is_flagged_as_bad(self) -> None:
        """The live shape: 35 of 90, interval clears 50 on the low side."""
        summary = clv_summary(self._record(35, 52, 3))
        self.assertEqual(summary["picks"], 90)
        self.assertTrue(summary["worseThanCoinFlip"])
        self.assertFalse(summary["beatsCoinFlip"])

    def test_an_ambiguous_rate_is_flagged_as_neither(self) -> None:
        """Both flags false is the honest answer for a thin sample."""
        summary = clv_summary(self._record(9, 11))
        self.assertFalse(summary["beatsCoinFlip"])
        self.assertFalse(summary["worseThanCoinFlip"])

    def test_a_good_rate_is_not_flagged_as_bad(self) -> None:
        summary = clv_summary(self._record(290, 210))
        self.assertTrue(summary["beatsCoinFlip"])
        self.assertFalse(summary["worseThanCoinFlip"])

    def test_an_unmoved_line_is_not_counted_as_a_defeat(self) -> None:
        """A line that never moved is a non-event, not a loss."""
        summary = clv_summary(self._record(5, 5, unmoved=10))
        self.assertEqual(summary["unmoved"], 10)
        self.assertEqual(summary["beatCloseP"], 25.0)  # 5 of 20, as computed
        self.assertEqual(summary["picks"], 20)

    def test_the_median_is_reported_and_survives_a_skewed_mean(self) -> None:
        """The exact trap in the live data: mean near zero, median clearly negative."""
        picks = [self._pick(-0.6)] * 9 + [self._pick(+12.0)]
        summary = clv_summary(picks)
        self.assertEqual(summary["medianPct"], -0.6)
        self.assertGreater(summary["avgPct"], 0, "mean is dragged positive by the outlier")
        self.assertLess(summary["medianPct"], 0, "median tells the truth")

    def test_leagues_are_split_because_they_disagree(self) -> None:
        """Pooled, MLB at 35% and WNBA at 56% average into one useless number."""
        picks = self._record(6, 20, league="mlb") + self._record(9, 7, league="wnba")
        summary = clv_summary(picks)
        self.assertEqual(summary["byLeague"]["mlb"]["picks"], 26)
        self.assertEqual(summary["byLeague"]["wnba"]["picks"], 16)
        self.assertTrue(summary["byLeague"]["mlb"]["worseThanCoinFlip"])
        self.assertFalse(summary["byLeague"]["wnba"]["worseThanCoinFlip"])

    def test_the_whole_record_is_the_headline_not_the_recent_window(self) -> None:
        """The window read 42.3% and spanned 50; the full record read 38.9% and did not."""
        everything = self._record(35, 55)
        recent = self._record(9, 11)
        summary = clv_summary(everything, recent)
        self.assertEqual(summary["picks"], 90)
        self.assertEqual(summary["last7Days"]["picks"], 20)
        self.assertTrue(summary["worseThanCoinFlip"])

    def test_an_empty_record_claims_nothing_either_way(self) -> None:
        summary = clv_summary([])
        self.assertIsNone(summary["worseThanCoinFlip"])
        self.assertIsNone(summary["medianPct"])
        self.assertEqual(summary["byLeague"], {})


class PriceHistoryTests(unittest.TestCase):
    """Keep the path between open and close, affordably.

    Line movement is among the most predictive public signals and every build
    was throwing it away: fetch a price, compare to nothing, overwrite. The
    feed is already paid for, so the path costs storage alone -- and unlike a
    feature derivable later from data on disk, history not recorded now is
    gone. That asymmetry is why this shipped before the analysis using it.

    The whole design rests on deduplication. Builds land roughly hourly and a
    baseball line is unchanged across most of them, so sampling every build
    would add thousands of identical rows a week to a file already committed
    on every run.
    """

    def _extend(self, existing, odds, at="t2", started=False):
        from accuracy_tracker import _extend_price_history
        return _extend_price_history(existing, odds, at, started=started)

    def test_the_first_price_is_recorded(self) -> None:
        self.assertEqual(self._extend(None, -120, "t1"), [{"at": "t1", "odds": -120}])

    def test_an_unchanged_price_is_not_recorded_again(self) -> None:
        """The affordability of the whole feature is this line."""
        history = [{"at": "t1", "odds": -120}]
        for _ in range(20):
            history = self._extend(history, -120)
        self.assertEqual(len(history), 1)

    def test_a_move_is_recorded(self) -> None:
        history = self._extend([{"at": "t1", "odds": -120}], -135, "t2")
        self.assertEqual([e["odds"] for e in history], [-120, -135])

    def test_a_move_back_is_recorded_as_its_own_observation(self) -> None:
        """Dedupe is against the previous price, not against every price seen."""
        history = [{"at": "t1", "odds": -120}]
        history = self._extend(history, -135, "t2")
        history = self._extend(history, -120, "t3")
        self.assertEqual([e["odds"] for e in history], [-120, -135, -120])

    def test_nothing_is_appended_once_the_game_starts(self) -> None:
        """An in-play quote is not part of the pre-game path."""
        history = [{"at": "t1", "odds": -120}]
        history = self._extend(history, +400, "t2", started=True)
        self.assertEqual([e["odds"] for e in history], [-120])

    def test_a_missing_price_does_not_append_a_hole(self) -> None:
        """A provider blip must not enter the series as a data point."""
        history = self._extend([{"at": "t1", "odds": -120}], None)
        self.assertEqual([e["odds"] for e in history], [-120])

    def test_the_series_is_capped(self) -> None:
        from accuracy_tracker import MAX_PRICE_OBSERVATIONS
        history = []
        for i in range(MAX_PRICE_OBSERVATIONS + 25):
            history = self._extend(history, -100 - i, f"t{i}")
        self.assertEqual(len(history), MAX_PRICE_OBSERVATIONS)

    def test_the_cap_keeps_the_most_recent_prices(self) -> None:
        """Truncating the wrong end would throw away the approach to the close."""
        from accuracy_tracker import MAX_PRICE_OBSERVATIONS
        history = []
        for i in range(MAX_PRICE_OBSERVATIONS + 5):
            history = self._extend(history, -100 - i, f"t{i}")
        self.assertEqual(history[-1]["odds"], -100 - (MAX_PRICE_OBSERVATIONS + 4))

    def test_a_corrupt_stored_series_does_not_take_the_build_down(self) -> None:
        """Same contract as every other reader of this file."""
        for junk in (None, [], ["not a dict"], [{"no": "odds"}], [{"odds": None}]):
            with self.subTest(junk=junk):
                out = self._extend(junk, -110, "t1")
                self.assertEqual(out[-1]["odds"], -110)

    def test_when_the_opening_price_was_taken_is_recorded(self) -> None:
        """`openingOddsAt` is what makes the timing hypothesis testable at all.

        An adverse CLV is what a side taken days early and drifting out looks
        like. Without a timestamp on the opening price there is no way to ask
        how far ahead the model committed, so the leading explanation for the
        38.9% beat rate could not be tested.
        """
        import accuracy_tracker
        source = Path(accuracy_tracker.__file__).read_text(encoding="utf-8")
        self.assertIn('"openingOddsAt": opening_at', source)
        self.assertIn('"priceHistory": price_history', source)


class BoardReportsNegativeClvTests(unittest.TestCase):
    """The page must say it, not just the JSON.

    The standing caveat quoted "+2.6% return across 874 graded picks" and
    nothing else. When the model's own better predictor of long-run profit
    points the other way, quoting only the return is selective.
    """

    @staticmethod
    def _board() -> str:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        return (root / "dashboard" / "board.js").read_text(encoding="utf-8")

    def test_the_caveat_reads_the_clv_flag(self) -> None:
        self.assertIn("worseThanCoinFlip", self._board())

    def test_the_card_can_say_worse_rather_than_only_inconclusive(self) -> None:
        self.assertIn("WORSE than a coin flip", self._board())

    def test_the_card_headlines_the_median_not_the_mean(self) -> None:
        self.assertIn("clv.medianPct", self._board())


if __name__ == "__main__":
    unittest.main()
