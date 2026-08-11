"""The deploy step cannot wait longer than the action allows.

On 2026-08-06 two builds differed only in how long GitHub's Pages queue took:
9m39s deployed, 10m02s hit "Timeout reached, aborting!" and was recorded as a
failure. The fix shipped for that was `timeout: 1200000` on the deploy step,
described in the workflow as twenty minutes of headroom.

It never had any. `actions/deploy-pages` caps the input, and every run since has
logged

    Warning: timeout value is greater than the allowed maximum -
    timeout set to the maximum of 600000 milliseconds

-- read off the 02:06Z run of 2026-08-11, which is how this was noticed. The
step was still waiting exactly ten minutes, and the workflow comment said
otherwise, which is the worse half of the bug: a mitigation everyone believed
was in place.

So the ceiling is a fact about the action, not a preference, and a second
attempt is the only headroom actually available. These tests hold both: the
value stays inside what the action will honour, and the retry stays wired up.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"

# The documented maximum for actions/deploy-pages. Anything above it is silently
# reduced to this, so a larger number in the file is not a longer wait -- it is
# a comment that disagrees with the run log.
MAX_DEPLOY_TIMEOUT_MS = 600_000


class DeployTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = WORKFLOW.read_text(encoding="utf-8")

    def _timeouts(self) -> list[int]:
        """Every `timeout:` given to a deploy-pages step, in milliseconds.

        Deliberately not `timeout-minutes:`, which is the job-level control and
        a different thing with a different unit -- the pattern requires the
        colon straight after the word.
        """
        return [int(value) for value in re.findall(r"^\s+timeout:\s*(\d+)\s*$", self.source, re.M)]

    def test_the_step_asks_for_a_timeout_at_all(self) -> None:
        self.assertTrue(self._timeouts(), "no deploy timeout found in pages.yml")

    def test_no_timeout_exceeds_what_the_action_will_honour(self) -> None:
        for value in self._timeouts():
            self.assertLessEqual(
                value,
                MAX_DEPLOY_TIMEOUT_MS,
                f"{value}ms is clamped to {MAX_DEPLOY_TIMEOUT_MS}ms at run time, so the "
                "workflow would claim a wait it does not get",
            )

    def test_a_timed_out_deploy_gets_a_second_attempt(self) -> None:
        """The only headroom available once the timeout cannot be raised."""
        self.assertIn("Retry the Pages deployment", self.source)
        self.assertIn("steps.deployment.outcome == 'failure'", self.source)

    def test_the_first_attempt_does_not_end_the_job(self) -> None:
        """Without this the retry is unreachable and the step is decoration."""
        deploy = self.source[self.source.index("- name: Deploy to GitHub Pages"):]
        deploy = deploy[: deploy.index("- name: Retry the Pages deployment")]
        self.assertIn("continue-on-error: true", deploy)

    def test_the_environment_url_survives_a_retry(self) -> None:
        """The first step publishes no page_url when it timed out."""
        self.assertIn("steps.deployment_retry.outputs.page_url", self.source)

    def test_the_job_allows_time_for_both_attempts(self) -> None:
        found = re.search(r"timeout-minutes:\s*(\d+)", self.source)
        self.assertIsNotNone(found, "the job has no timeout-minutes")
        minutes = int(found.group(1))
        # Two ten-minute waits plus the ~5 minutes the build itself takes.
        self.assertGreaterEqual(minutes, 25, "a retry would be cut off by the job timeout")

    def test_the_workflow_records_why_the_timeout_is_not_raised_again(self) -> None:
        """The comment claiming twenty minutes is what made this invisible."""
        self.assertNotIn("timeout: 1200000", self.source)
        self.assertIn("allowed maximum", self.source)


if __name__ == "__main__":
    unittest.main()
