"""The accuracy page must lead with what predicts, not what flatters.

Reviewed 13 Aug 2026. The page was good in structure -- the verdict panel
especially -- and consistently ordered against the reader: the headline strip
opened with hit rate and return, both outcome statistics over a mixed
population, while closing line value sat in a card further down. CLV is the
better predictor of long-run profit and it is currently negative, so a reader
scanning the strip took away "61.8%, +2.6%" and stopped.

Four of the seven changes here are the same idea in four places: a pooled or
blended number standing where a population-specific one belongs. That defect
has now appeared five times in this project -- totals, spreads, the staking
gate, closing line value, divergence -- so these are tests about a pattern
rather than about a page.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BOARD_JS = ROOT / "dashboard" / "board.js"


class HeadlineStripTests(unittest.TestCase):
    """What a reader sees before they read anything."""

    def setUp(self) -> None:
        source = BOARD_JS.read_text(encoding="utf-8")
        start = source.index('readings($("#accReadings")')
        self.strip = source[start:source.index("]);", start)]

    def test_closing_line_value_comes_first(self) -> None:
        self.assertLess(
            self.strip.index("Closing line value"), self.strip.index("Hit rate"),
            "CLV predicts long-run profit better than the hit rate does, and the "
            "verdict panel already tells readers to weigh it first",
        )

    def test_the_hit_rate_shown_is_the_priced_one(self) -> None:
        """`allTime.pct` includes picks that never carried a price, which is not
        the population any break-even or return applies to."""
        self.assertIn("priced.pct", self.strip)

    def test_the_priced_sample_size_is_named_beside_it(self) -> None:
        self.assertIn("priced)", self.strip)


class PricedPopulationTests(unittest.TestCase):
    """The blended/priced split, applied to the biggest number on the page."""

    def test_the_tracker_publishes_a_priced_all_time_record(self) -> None:
        source = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        self.assertIn('"allTimePriced": priced_all_time', source)

    def test_only_picks_with_a_price_enter_it(self) -> None:
        source = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        self.assertIn('if item.get("pickOdds") is not None:', source)

    def test_it_is_computed_over_the_real_record(self) -> None:
        from accuracy_tracker import _accumulate_summary, _summary_bucket

        bucket = _summary_bucket()
        for item in (
            {"status": "graded", "correct": True, "units": 0.9},
            {"status": "graded", "correct": False, "units": -1.0},
        ):
            _accumulate_summary(bucket, item)
        self.assertEqual(bucket["total"], 2)
        self.assertEqual(bucket["pct"], 50.0)


class KeyCollisionTests(unittest.TestCase):
    """`pricedPct` meant two different things in one file.

    In `byLeague` it was the share of picks carrying a price -- MLB 92.4. In the
    market summaries it was the hit rate among priced picks -- totals 52.3. Two
    meanings, one name, and the staking-gate bug fixed the day before was
    exactly this class of confusion.
    """

    def test_the_ambiguous_name_is_gone_from_the_source(self) -> None:
        for name in ("accuracy_tracker.py", "mlb_predictions.py"):
            source = (ROOT / name).read_text(encoding="utf-8")
            code = "\n".join(
                line for line in source.split("\n") if not line.lstrip().startswith("#")
            )
            self.assertNotIn('"pricedPct"', code, f"{name} still writes the ambiguous key")

    def test_coverage_and_hit_rate_have_separate_names(self) -> None:
        tracker = (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")
        self.assertIn('"pricedSharePct"', tracker)
        self.assertIn('"pricedHitPct"', tracker)

    def test_the_gate_reads_the_hit_rate_one(self) -> None:
        """Reading the coverage share here would stake on the wrong number."""
        source = (ROOT / "mlb_predictions.py").read_text(encoding="utf-8")
        self.assertIn('record.get("pricedHitPct")', source)

    def test_the_page_reads_each_under_its_own_name(self) -> None:
        board = BOARD_JS.read_text(encoding="utf-8")
        self.assertIn("v.pricedSharePct", board)
        self.assertIn("m.pricedHitPct", board)


class ContaminationNoticeTests(unittest.TestCase):
    """The page presented leaked figures as trustworthy.

    Log loss, Brier, AUC and reliability are all fitted on rows whose features
    were recomputed after the result was known. Saying so is not optional while
    it is true.
    """

    def setUp(self) -> None:
        self.board = BOARD_JS.read_text(encoding="utf-8")

    def test_the_notice_exists_and_is_rendered(self) -> None:
        self.assertIn("function contaminationNote(", self.board)
        self.assertIn("contaminationNote(S.ablation)", self.board)

    def test_it_lifts_itself_once_the_clean_sample_is_big_enough(self) -> None:
        """A warning that never clears is one nobody reads."""
        self.assertIn("frozen >= CLEAN_SAMPLE_TARGET", self.board)

    def test_it_quotes_the_measurable_evidence(self) -> None:
        """0.682 against the market's 0.640 is the part that is not an opinion."""
        self.assertIn("0.682", self.board)

    def test_it_says_the_picks_themselves_were_not_affected(self) -> None:
        """True, and the difference matters to anyone reading the board."""
        self.assertIn("at pick time the game has not been played", self.board)


class LeagueDivergenceTests(unittest.TestCase):
    """The pooled gap read 3.7pts while NFL preseason sat at 27.2."""

    def setUp(self) -> None:
        self.board = BOARD_JS.read_text(encoding="utf-8")

    def test_the_league_table_shows_the_gap_to_market(self) -> None:
        self.assertIn("Gap to market", self.board)
        self.assertIn("medianGapPct", self.board)

    def test_a_wild_league_is_flagged_rather_than_merely_listed(self) -> None:
        self.assertIn("inputs suspect", self.board)

    def test_the_flag_ignores_thin_samples(self) -> None:
        """One preseason game can post a 40pt median honestly."""
        self.assertIn("gapN >= 10", self.board)

    def test_the_evaluation_supplies_the_data(self) -> None:
        source = (ROOT / "scripts" / "evaluation.py").read_text(encoding="utf-8")
        self.assertIn('report["byLeague"]', source)


class ClvTrendTests(unittest.TestCase):
    """Everything else on the page is all-time or last-seven-days.

    While CLV is negative the only question worth answering is whether it is
    improving, and the page could not answer it.
    """

    def setUp(self) -> None:
        self.board = BOARD_JS.read_text(encoding="utf-8")

    def test_the_panel_exists_and_is_rendered(self) -> None:
        self.assertIn("function clvTrendPanel(", self.board)
        self.assertIn("clvTrendPanel(A)", self.board)

    def test_it_plots_a_rolling_median_not_raw_picks(self) -> None:
        """Single-pick CLV is noise, and the headline figure is a median too,
        so the line and the number should be measuring the same thing."""
        self.assertIn("CLV_WINDOW", self.board)
        self.assertIn("rolling median", self.board)

    def test_it_only_uses_confirmed_closes(self) -> None:
        self.assertIn("r.pickOddsFrozenAt", self.board)

    def test_it_declines_to_draw_a_line_through_too_little_data(self) -> None:
        self.assertIn("picks.length < CLV_WINDOW * 2", self.board)

    def test_it_draws_the_break_even_reference(self) -> None:
        """A CLV chart with no zero line cannot be read at a glance."""
        self.assertIn("stroke-dasharray", self.board)

    def test_the_chart_carries_a_text_alternative(self) -> None:
        self.assertIn('role="img"', self.board)
        self.assertIn("aria-label", self.board)


class SideMarketStanceTests(unittest.TestCase):
    """Totals and spreads are no longer staked; the page said nothing."""

    def test_the_card_states_the_stance(self) -> None:
        self.assertIn("Published, not staked", BOARD_JS.read_text(encoding="utf-8"))

    def test_the_stance_is_derived_from_the_same_rule_as_the_gate(self) -> None:
        """Hardcoding "not staked" would drift the moment a record earned it."""
        board = BOARD_JS.read_text(encoding="utf-8")
        self.assertIn("pricedHit - m.pricedStdErrPct > m.breakEvenPct", board)


class LivePayloadTests(unittest.TestCase):
    """The page reads keys the tracker actually writes."""

    def test_the_committed_accuracy_file_has_what_the_page_expects(self) -> None:
        path = ROOT / "docs" / "data" / "accuracy.json"
        if not path.is_file():
            self.skipTest("no committed accuracy.json")
        summary = json.loads(path.read_text(encoding="utf-8")).get("summary") or {}
        leagues = list((summary.get("byLeague") or {}).values())
        # Tolerates a file written before the rename -- it regenerates on the
        # next build. What it will not tolerate is both names at once, which
        # would mean a half-finished migration with two sources of truth.
        for league in leagues:
            if "pricedSharePct" in league:
                self.assertNotIn("pricedPct", league, "both names present: migration is half done")
        for market in ("totals", "spreads"):
            record = summary.get(market) or {}
            if "pricedHitPct" in record:
                self.assertNotIn("pricedPct", record, "both names present: migration is half done")


if __name__ == "__main__":
    unittest.main()
