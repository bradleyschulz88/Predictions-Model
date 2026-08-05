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
    script = f"{match.group(0)}\nconsole.log(JSON.stringify(buildAgeNote({arg})));"
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
        script = match.group(0) + '\nconsole.log(JSON.stringify(buildAgeNote("not a date")));'
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


if __name__ == "__main__":
    unittest.main()
