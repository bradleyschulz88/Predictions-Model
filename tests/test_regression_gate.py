"""The gate that decides whether every build ships had no tests at all.

`check_regression.py` is the last thing between a degraded model and the
published board, and it is also the thing most able to stop a healthy build.
Both halves of that matter and neither was covered. Two of its checks failed
scheduled runs this month -- the intercept ceiling on 2026-08-22 -- and in each
case the gate was right about the model and nobody could tell without reading
the source.

Two failures worth designing against, both seen for real:

**A gate that only speaks when it fails.** "OK: within tolerance" reads
identically at 5% of the budget and at 95%, so a slow drift is invisible until
the day it is fatal. The margin is now printed on every run, and a warning
fires at 60%.

**A one-way ratchet across incomparable samples.** The baseline tightens
whenever the score improves, regardless of how many games that score came from.
Walk-forward log loss on 400 games is not the same measurement as on 950, so a
lucky small run could lock in a bar the honest larger sample never clears
again -- a permanent build failure with no bug behind it. Live numbers say this
is the direction of travel: the baseline was recorded at n=810 and the log is
now at n=950, with 19% of the log-loss budget already spent.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import check_regression as gate  # noqa: E402


def _run(*, current: dict, baseline: dict | None, update: bool = False) -> tuple[int, str, dict | None]:
    """Drive check() against files in a temp dir, returning what it wrote."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        evaluation = directory / "evaluation.json"
        baseline_file = directory / "model_baseline.json"
        weights = directory / "model_weights.json"

        evaluation.write_text(json.dumps({"fittedWalkForward": current}), encoding="utf-8")
        weights.write_text(json.dumps({"leagueIntercepts": {}}), encoding="utf-8")
        if baseline is not None:
            baseline_file.write_text(json.dumps(baseline), encoding="utf-8")

        buffer = io.StringIO()
        with mock.patch.object(gate, "EVALUATION_FILE", evaluation), \
             mock.patch.object(gate, "BASELINE_FILE", baseline_file), \
             mock.patch.object(gate, "WEIGHTS_FILE", weights), \
             redirect_stdout(buffer):
            code = gate.check(update=update)

        written = json.loads(baseline_file.read_text(encoding="utf-8")) if baseline_file.is_file() else None
        return code, buffer.getvalue(), written


GOOD = {"logLoss": 0.6354, "brier": 0.2237, "n": 810}


class PassAndFailTests(unittest.TestCase):
    def test_a_matching_score_passes(self) -> None:
        code, _out, _written = _run(current=dict(GOOD), baseline=dict(GOOD))
        self.assertEqual(code, 0)

    def test_a_score_inside_tolerance_passes(self) -> None:
        current = {"logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE / 2, "brier": 0.2237, "n": 900}
        code, _out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(code, 0)

    def test_a_log_loss_regression_fails(self) -> None:
        current = {"logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE + 0.001, "brier": 0.2237, "n": 900}
        code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(code, 1)
        self.assertIn("log loss", out)

    def test_a_brier_regression_fails(self) -> None:
        current = {"logLoss": 0.6354, "brier": 0.2237 + gate.BRIER_TOLERANCE + 0.001, "n": 900}
        code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(code, 1)
        self.assertIn("brier", out)

    def test_a_coin_flip_fails_however_good_the_baseline_is(self) -> None:
        """A broken model is broken regardless of what it is measured against."""
        current = {"logLoss": gate.COIN_FLIP_LOG_LOSS, "brier": 0.25, "n": 900}
        code, out, _written = _run(current=current, baseline={"logLoss": 0.99, "brier": 0.99, "n": 10})
        self.assertEqual(code, 1)
        self.assertIn("coin flip", out)

    def test_a_first_run_with_no_baseline_writes_one(self) -> None:
        code, out, written = _run(current=dict(GOOD), baseline=None)
        self.assertEqual(code, 0)
        self.assertIn("Wrote baseline", out)
        self.assertEqual(written["logLoss"], GOOD["logLoss"])


class VisibleMarginTests(unittest.TestCase):
    """A gate that says nothing until it fails gives no chance to act."""

    def test_a_passing_run_reports_how_much_budget_is_left(self) -> None:
        current = {"logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE / 2, "brier": 0.2237, "n": 900}
        _code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertIn("budget used", out)
        self.assertIn("headroom", out)

    def test_the_reported_percentage_tracks_the_real_margin(self) -> None:
        current = {"logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE * 0.5, "brier": 0.2237, "n": 900}
        _code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertIn("logloss 50%", out)

    def test_a_score_better_than_baseline_reports_zero_not_a_negative(self) -> None:
        current = {"logLoss": 0.60, "brier": 0.20, "n": 900}
        _code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertIn("logloss 0%", out)
        self.assertNotIn("-", out.split("budget used")[1].split("\n")[0])

    def test_drifting_past_the_warn_line_says_so(self) -> None:
        current = {
            "logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE * (gate.BUDGET_WARN_PCT + 10) / 100,
            "brier": 0.2237,
            "n": 900,
        }
        _code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertIn("::warning::", out)
        self.assertIn("regression budget", out)

    def test_a_comfortable_run_does_not_cry_wolf(self) -> None:
        current = {"logLoss": 0.6354 + gate.LOG_LOSS_TOLERANCE * 0.1, "brier": 0.2237, "n": 900}
        _code, out, _written = _run(current=current, baseline=dict(GOOD))
        self.assertNotIn("::warning::", out)


class RatchetTests(unittest.TestCase):
    """One-way, but only between things that can be compared."""

    def test_an_improvement_on_more_data_tightens_the_baseline(self) -> None:
        current = {"logLoss": 0.6300, "brier": 0.2200, "n": 950}
        _code, out, written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(written["logLoss"], 0.6300)
        self.assertEqual(written["n"], 950)
        self.assertIn("ratcheted", out)

    def test_an_improvement_on_the_same_amount_of_data_tightens_it(self) -> None:
        current = {"logLoss": 0.6300, "brier": 0.2200, "n": 810}
        _code, _out, written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(written["logLoss"], 0.6300)

    def test_an_improvement_on_less_data_does_not(self) -> None:
        """The bug this guard exists for: a lucky small run permanently raising
        a bar the honest larger sample may never clear again."""
        current = {"logLoss": 0.5000, "brier": 0.1500, "n": 100}
        _code, out, written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(written["logLoss"], GOOD["logLoss"], "baseline was moved by a small sample")
        self.assertIn("smaller sample", out)

    def test_the_baseline_records_the_sample_it_came_from(self) -> None:
        """Without n stored there is nothing to compare a later run against."""
        current = {"logLoss": 0.6300, "brier": 0.2200, "n": 950}
        _code, _out, written = _run(current=current, baseline=dict(GOOD))
        self.assertIn("n", written)

    def test_a_worse_score_never_loosens_the_baseline(self) -> None:
        current = {"logLoss": 0.6400, "brier": 0.2300, "n": 5000}
        _code, _out, written = _run(current=current, baseline=dict(GOOD))
        self.assertEqual(written["logLoss"], GOOD["logLoss"])

    def test_update_baseline_accepts_a_worse_but_working_model(self) -> None:
        """The documented escape hatch when a regression is intended."""
        current = {"logLoss": 0.6800, "brier": 0.2400, "n": 900}
        code, out, written = _run(current=current, baseline=dict(GOOD), update=True)
        self.assertEqual(code, 0)
        self.assertEqual(written["logLoss"], 0.6800)
        self.assertIn("Wrote baseline", out)

    def test_update_baseline_still_refuses_a_broken_model(self) -> None:
        """The escape hatch is for a judgement call about quality, not a way to
        enshrine a model that cannot beat a coin flip. The coin-flip check runs
        first and is not overridable, which is correct -- found by writing this
        test expecting the opposite."""
        current = {"logLoss": 0.9, "brier": 0.4, "n": 10}
        code, out, _written = _run(current=current, baseline=dict(GOOD), update=True)
        self.assertEqual(code, 1)
        self.assertIn("coin flip", out)


class InterceptCheckTests(unittest.TestCase):
    """The check that actually failed 30 builds."""

    def test_an_intercept_past_the_ceiling_is_reported(self) -> None:
        failures = gate.check_intercepts({"leagueIntercepts": {"nfl": -0.2722}})
        self.assertTrue(failures)
        self.assertIn("nfl", failures[0])

    def test_an_intercept_inside_it_is_not(self) -> None:
        self.assertEqual(gate.check_intercepts({"leagueIntercepts": {"nfl": -0.1506}}), [])

    def test_the_ceiling_is_symmetric(self) -> None:
        """A large positive correction is the same claim as a large negative one."""
        over = gate.MAX_RESIDUAL_INTERCEPT + 0.01
        self.assertTrue(gate.check_intercepts({"leagueIntercepts": {"a": over}}))
        self.assertTrue(gate.check_intercepts({"leagueIntercepts": {"a": -over}}))

    def test_missing_weights_do_not_crash_the_gate(self) -> None:
        self.assertEqual(gate.check_intercepts({}), [])
        self.assertEqual(gate.check_intercepts({"leagueIntercepts": {}}), [])

    def test_the_live_weights_pass(self) -> None:
        """Against whatever the refit last wrote, if anything did."""
        if not gate.WEIGHTS_FILE.is_file():
            self.skipTest("no model_weights.json (gitignored, written by the refit)")
        self.assertEqual(gate.check_intercepts(), [])


class DivergenceCheckTests(unittest.TestCase):
    # The gate reads `divergence.byLeague`, not a top-level `byLeague` --
    # evaluation.json has both, and they are different blocks. Writing these
    # against the wrong one produced a check that silently passed everything,
    # which is exactly the failure this file is meant to prevent.
    @staticmethod
    def _report(by_league: dict) -> dict:
        return {"divergence": {"byLeague": by_league}}

    def test_a_wild_league_on_enough_games_fails(self) -> None:
        failures = gate.check_divergence(self._report({"nfl": {"medianGapPct": 27.2, "n": 40}}))
        self.assertTrue(failures)
        self.assertIn("nfl", failures[0])

    def test_a_wild_league_on_too_few_games_does_not(self) -> None:
        """One preseason game can post a 40pt median honestly."""
        report = self._report({"nfl": {"medianGapPct": 40.0, "n": gate.MIN_DIVERGENCE_SAMPLE - 1}})
        self.assertEqual(gate.check_divergence(report), [])

    def test_a_normal_league_passes(self) -> None:
        self.assertEqual(gate.check_divergence(self._report({"mlb": {"medianGapPct": 8.2, "n": 600}})), [])

    def test_an_empty_report_does_not_crash(self) -> None:
        self.assertEqual(gate.check_divergence({}), [])
        self.assertEqual(gate.check_divergence({"divergence": {}}), [])

    def test_the_block_it_reads_is_the_one_the_evaluation_writes(self) -> None:
        """A gate reading a key that does not exist passes everything forever,
        and this project has shipped that bug twice under other names."""
        path = ROOT / "docs" / "data" / "evaluation.json"
        if not path.is_file():
            self.skipTest("no committed evaluation.json")
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("divergence", report)
        self.assertIn("byLeague", report["divergence"])
        self.assertTrue(report["divergence"]["byLeague"], "the block the gate reads is empty")

    def test_the_live_report_passes(self) -> None:
        path = ROOT / "docs" / "data" / "evaluation.json"
        if not path.is_file():
            self.skipTest("no committed evaluation.json")
        self.assertEqual(gate.check_divergence(json.loads(path.read_text(encoding="utf-8"))), [])


if __name__ == "__main__":
    unittest.main()
