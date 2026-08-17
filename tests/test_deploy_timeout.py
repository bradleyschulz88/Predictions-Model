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

    def test_every_retry_waits_before_it_runs(self) -> None:
        """A retry with no delay is one attempt with extra logging.

        The original pair fired 1.75 seconds apart against a Pages API outage
        that lasted over two hours. The comment reasoned about timeouts, where
        ten minutes have already elapsed -- but a 503 answers immediately, so
        nothing had changed by the time the retry ran.
        """
        for attempt, previous in (
            ("Retry the Pages deployment", "Wait before the first retry"),
            ("Last retry of the Pages deployment", "Wait before the last retry"),
        ):
            self.assertIn(previous, self.source, f"{attempt} has no wait before it")
            self.assertLess(
                self.source.index(previous), self.source.index(f"- name: {attempt}"),
                f"{previous} must come before {attempt}",
            )

    def test_the_waits_escalate(self) -> None:
        sleeps = [int(v) for v in re.findall(r"^\s+run: sleep (\d+)\s*$", self.source, re.M)]
        self.assertEqual(len(sleeps), 2, f"expected two backoff waits, found {sleeps}")
        self.assertGreater(sleeps[1], sleeps[0], "the second wait should be longer")
        self.assertGreaterEqual(sleeps[0], 30, "too short to outlast a transient outage")

    def test_the_middle_attempt_does_not_end_the_job(self) -> None:
        """Without this the last retry is unreachable, which is the same bug
        the first attempt already had `continue-on-error` for."""
        block = self.source[self.source.index("- name: Retry the Pages deployment"):]
        block = block[: block.index("- name: Wait before the last retry")]
        self.assertIn("continue-on-error: true", block)

    def test_the_final_attempt_is_allowed_to_fail_the_build(self) -> None:
        """Three attempts across three minutes is enough to mean something."""
        block = self.source[self.source.index("- name: Last retry of the Pages deployment"):]
        self.assertNotIn("continue-on-error", block)

    def test_the_last_retry_only_runs_after_both_others_failed(self) -> None:
        self.assertIn(
            "steps.deployment.outcome == 'failure' && steps.deployment_retry.outcome == 'failure'",
            self.source,
        )

    def test_the_waits_cost_nothing_on_a_healthy_build(self) -> None:
        """Each sleep is gated on a failure, so the normal path never waits."""
        for line_no, line in enumerate(self.source.split("\n")):
            if not re.match(r"^\s+run: sleep \d+\s*$", line):
                continue
            window = "\n".join(self.source.split("\n")[max(0, line_no - 4):line_no])
            self.assertIn("outcome == 'failure'", window, f"unguarded sleep at line {line_no + 1}")

    def test_the_first_attempt_does_not_end_the_job(self) -> None:
        """Without this the retry is unreachable and the step is decoration."""
        deploy = self.source[self.source.index("- name: Deploy to GitHub Pages"):]
        deploy = deploy[: deploy.index("- name: Retry the Pages deployment")]
        self.assertIn("continue-on-error: true", deploy)

    def test_the_environment_url_survives_a_retry(self) -> None:
        """The first step publishes no page_url when it timed out."""
        self.assertIn("steps.deployment_retry.outputs.page_url", self.source)

    def test_the_job_allows_time_for_every_attempt(self) -> None:
        found = re.search(r"timeout-minutes:\s*(\d+)", self.source)
        self.assertIsNotNone(found, "the job has no timeout-minutes")
        minutes = int(found.group(1))
        sleeps = sum(int(v) for v in re.findall(r"^\s+run: sleep (\d+)\s*$", self.source, re.M))
        attempts = len(self._timeouts())
        # Worst case: every attempt runs to its ceiling, every backoff elapses,
        # and the build itself takes the ~8 minutes its slowest recent run did.
        worst_case = attempts * (MAX_DEPLOY_TIMEOUT_MS / 60_000) + sleeps / 60 + 8
        self.assertGreaterEqual(
            minutes, worst_case,
            f"{attempts} attempts plus {sleeps}s of backoff needs {worst_case:.0f} min, job allows {minutes}",
        )

    def test_the_workflow_records_why_the_timeout_is_not_raised_again(self) -> None:
        """The comment claiming twenty minutes is what made this invisible."""
        self.assertNotIn("timeout: 1200000", self.source)
        self.assertIn("allowed maximum", self.source)


if __name__ == "__main__":
    unittest.main()
