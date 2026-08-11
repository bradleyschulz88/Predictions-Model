"""The page reports how old its data is instead of promising a cadence.

Two strings told readers that "Predictions refresh every 30 minutes on GitHub
Actions". The cron does ask for that, and GitHub does not deliver it. Measured
2026-08-05, gaps between scheduled runs were 83, 99, 109, 109, 123, 136, 170,
194 and 266 minutes -- averaging about two hours against a nominal thirty.

No run was cancelled and none failed; the scheduler simply does not create
them, which is documented best-effort behaviour for `schedule`. The cron had
already been moved off the congested :00/:30 to :07/:37 for exactly this
reason and it changed nothing measurable.

So a reader comparing a board built two hours ago against a promise of thirty
minutes would reasonably conclude something was broken. These pin the honest
version: report the age, and explain the gap only once it is large enough to
look wrong.

The function is lifted out of dashboard/app.js and run, rather than restated,
because a copy in the test would agree with itself if the thresholds drifted.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_JS = ROOT / "dashboard" / "app.js"
BLOCK = re.compile(r"^function buildAgeNote\(builtAt\) \{.*?^\}$", re.S | re.M)


def _node() -> str | None:
    return shutil.which("node")


def _note(minutes_old: float | None) -> str:
    """Run the real buildAgeNote against a timestamp this many minutes back."""
    match = BLOCK.search(APP_JS.read_text(encoding="utf-8"))
    assert match, "could not find buildAgeNote in app.js"
    arg = (
        "undefined" if minutes_old is None
        else f"new Date(Date.now() - {minutes_old} * 60000).toISOString()"
    )
    # The constant lives outside the extracted function now, so the harness
    # has to supply it. ThresholdAgreementTests is what keeps this literal
    # honest against the real one.
    script = (
        "const STALE_AFTER_MINUTES = 150;\n"
        + match.group(0)
        + f"\nconsole.log(JSON.stringify(buildAgeNote({arg})));"
    )
    out = subprocess.run(
        [_node(), "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


@unittest.skipUnless(_node(), "node is required to execute the dashboard helper")
class BuildAgeNoteTests(unittest.TestCase):
    def test_a_fresh_build_reads_as_fresh(self) -> None:
        self.assertIn("just now", _note(0))

    def test_minutes_are_reported_in_minutes(self) -> None:
        self.assertIn("17 minutes ago", _note(17))

    def test_hours_are_reported_in_hours(self) -> None:
        self.assertIn("hours ago", _note(300))

    def test_days_are_reported_in_days(self) -> None:
        self.assertIn("days ago", _note(60 * 24 * 2))

    def test_a_normal_gap_carries_no_excuse(self) -> None:
        """Under about two hours is the working range; explaining it is noise."""
        for minutes in (5, 45, 90):
            self.assertNotIn("best-effort", _note(minutes), f"{minutes} minutes")

    def test_a_long_gap_explains_itself(self) -> None:
        """This is the case that used to read as a broken build."""
        note = _note(200)
        self.assertIn("best-effort", note)
        self.assertIn("rather than a fault", note)

    def test_a_missing_timestamp_does_not_render_nonsense(self) -> None:
        """No NaN, no "Invalid Date", no promise it cannot keep."""
        note = _note(None)
        for bad in ("NaN", "Invalid", "undefined", "30 minutes"):
            self.assertNotIn(bad, note)

    def test_an_unparseable_timestamp_is_treated_as_missing(self) -> None:
        match = BLOCK.search(APP_JS.read_text(encoding="utf-8"))
        script = ("const STALE_AFTER_MINUTES = 150;\n" + match.group(0)
                  + '\nconsole.log(JSON.stringify(buildAgeNote("not a date")));')
        out = subprocess.run(
            [_node(), "-e", script], capture_output=True, text=True, timeout=30, check=True
        )
        self.assertNotIn("NaN", json.loads(out.stdout))


class CadencePromiseRemovedTests(unittest.TestCase):
    """Static, so it holds where node is unavailable."""

    def test_the_page_no_longer_promises_thirty_minutes(self) -> None:
        source = APP_JS.read_text(encoding="utf-8")
        self.assertNotIn("refresh every 30 minutes on GitHub Actions", source)

    def test_the_live_score_cadence_is_still_stated(self) -> None:
        """That one is a browser timer this project controls, so it is true."""
        self.assertIn("liveScoreRefreshSeconds", APP_JS.read_text(encoding="utf-8"))

    def test_the_workflow_records_why_the_cron_is_not_retuned(self) -> None:
        """It has been tuned once already for no measurable gain."""
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertIn("do not tune this again", workflow.lower())


BOARD_JS = ROOT / "dashboard" / "board.js"
BOARD_BLOCK = re.compile(r"^function buildAge\(updatedAt\) \{.*?^\}$", re.S | re.M)


def _board_age(minutes_old: float | None) -> str:
    """Run the real buildAge out of board.js, the always-visible footer one."""
    match = BOARD_BLOCK.search(BOARD_JS.read_text(encoding="utf-8"))
    assert match, "could not find buildAge in board.js"
    arg = (
        "undefined" if minutes_old is None
        else f"new Date(Date.now() - {minutes_old} * 60000).toISOString()"
    )
    script = (
        "const STALE_AFTER_MINUTES = 150;\n"
        + match.group(0)
        + f"\nconsole.log(JSON.stringify(buildAge({arg})));"
    )
    out = subprocess.run(
        [_node(), "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


@unittest.skipUnless(_node(), "node is required to execute the dashboard helper")
class FooterAgeTests(unittest.TestCase):
    """The front page footer has to say how old it is, not just when it built.

    This is the line a reader actually sees. buildAgeNote in app.js only
    renders inside a banner that appears when the board is already degraded --
    live fallback, a missing snapshot, or no games -- so on a healthy page it
    never shows at all. A browser check of the live site on 2026-08-07 found
    no age text anywhere, which was correct: there was none.

    Meanwhile a GitHub incident had left the board seven hours stale the day
    before, and the footer read "Built 6 Aug, 15:04" -- indistinguishable from
    a fresh build unless the reader does the subtraction.
    """

    def test_a_fresh_build_shows_its_age(self) -> None:
        self.assertIn("just now", _board_age(0))

    def test_minutes_then_hours_then_days(self) -> None:
        self.assertIn("40 min ago", _board_age(40))
        self.assertIn("hr ago", _board_age(60 * 5))
        self.assertIn("days ago", _board_age(60 * 24 * 3))

    def test_a_normal_gap_is_not_flagged(self) -> None:
        """GitHub delivers roughly every two hours; that is not a fault."""
        for minutes in (5, 60, 120):
            self.assertNotIn("stale", _board_age(minutes), f"{minutes} minutes")

    def test_a_stalled_build_is_flagged_visibly(self) -> None:
        """Seven hours is what the outage looked like."""
        note = _board_age(60 * 7)
        self.assertIn("stale", note)
        self.assertIn("not landing", note)

    def test_a_missing_timestamp_adds_nothing(self) -> None:
        """No NaN, no "Invalid Date" -- the absolute time still stands alone."""
        self.assertEqual(_board_age(None), "")

    def test_the_flag_has_a_style_to_render_with(self) -> None:
        css = (ROOT / "dashboard" / "board.css").read_text(encoding="utf-8")
        self.assertIn(".railfoot .stale", css)


class ThresholdAgreementTests(unittest.TestCase):
    """Both pages must call the same age stale.

    They are separate bundles with no shared module, so the constant is
    duplicated and nothing but this test would notice it drifting.
    """

    def _threshold(self, name: str) -> int:
        source = (ROOT / "dashboard" / name).read_text(encoding="utf-8")
        found = re.search(r"const STALE_AFTER_MINUTES = (\d+);", source)
        self.assertIsNotNone(found, f"{name} has no STALE_AFTER_MINUTES")
        return int(found.group(1))

    def test_board_and_tools_agree(self) -> None:
        self.assertEqual(self._threshold("board.js"), self._threshold("app.js"))

    def test_neither_page_hardcodes_the_number_instead(self) -> None:
        for name in ("board.js", "app.js"):
            source = (ROOT / "dashboard" / name).read_text(encoding="utf-8")
            self.assertNotIn("minutes >= 150", source, name)
            self.assertNotIn("minutes < 150", source, name)


CAVEAT_BLOCK = re.compile(r"^function standingCaveat\(accuracy\) \{.*?^\}$", re.S | re.M)


def _caveat(accuracy) -> str:
    """Run the real standingCaveat out of board.js."""
    match = CAVEAT_BLOCK.search(BOARD_JS.read_text(encoding="utf-8"))
    assert match, "could not find standingCaveat in board.js"
    script = (
        match.group(0)
        + f"\nconsole.log(JSON.stringify(standingCaveat({json.dumps(accuracy)})));"
    )
    out = subprocess.run(
        [_node(), "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


@unittest.skipUnless(_node(), "node is required to execute the dashboard helper")
class StandingCaveatTests(unittest.TestCase):
    """A page that prints a stake size has to say what it is.

    The board names picks "worth backing" and puts a Kelly stake against them,
    and said nothing about what that rests on. A stake reads as a
    recommendation, and the record behind it is thinner than the presentation:
    measured 2026-08-07, +2.0% over 802 graded picks overall, while totals ran
    48.6% on 70 priced picks and the whole spread record rested on 13.

    The figures are read from accuracy.json rather than written into the
    markup, so the line cannot drift into flattering the model.
    """

    def _full(self, total, roi):
        return {"summary": {"allTime": {"total": total, "roiPct": roi}}}

    def test_it_says_what_the_output_is(self) -> None:
        self.assertIn("not betting advice", _caveat(self._full(802, 2.0)))

    def test_it_quotes_the_real_return_and_sample(self) -> None:
        note = _caveat(self._full(802, 2.0))
        self.assertIn("802 graded picks", note)
        self.assertIn("+2% return", note)

    def test_a_losing_record_is_stated_as_losing(self) -> None:
        """The number is read from the data, so it cannot be quietly flattering."""
        self.assertIn("-3.4% return", _caveat(self._full(120, -3.4)))

    def test_it_warns_that_individual_markets_are_thinner(self) -> None:
        """Totals at 48.6% on 70 picks sit behind that headline."""
        self.assertIn("thinner", _caveat(self._full(802, 2.0)))

    def test_a_missing_roi_still_reads_as_a_sentence(self) -> None:
        note = _caveat({"summary": {"allTime": {"total": 40}}})
        self.assertIn("40 graded picks", note)
        self.assertNotIn("showing measured", note)
        self.assertNotIn("showing across", note)

    def test_no_record_falls_back_without_inventing_numbers(self) -> None:
        for payload in ({}, None, {"summary": {}}):
            note = _caveat(payload)
            self.assertIn("not betting advice", note)
            self.assertNotIn("undefined", note)
            self.assertNotIn("NaN", note)
            self.assertNotIn("graded picks", note)

    def test_the_footer_actually_renders_it(self) -> None:
        """Wired in, not just defined -- the mistake this file already caught once."""
        source = BOARD_JS.read_text(encoding="utf-8")
        self.assertIn("standingCaveat(accuracy)", source)


if __name__ == "__main__":
    unittest.main()
