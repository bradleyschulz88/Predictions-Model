"""A non-finite float does not print a dash on the page. It takes the page down.

`json.dumps` emits the bare literals `NaN`, `Infinity` and `-Infinity` by
default. None of the three is JSON, and `JSON.parse` throws on the first one it
meets, so nothing after the throw runs -- the reader gets a blank panel, not a
missing number.

Found 17 Aug 2026 by fuzzing `_accumulate_summary`, which summed
`float(item.get("units") or 0.0)` over stored rows. One NaN poisons the bucket
total, then the ROI derived from it, then `accuracy.json`, then the whole
accuracy page:

    bucket over 7 rows, one with units=NaN
      -> total=5  pct=60.0  units=nan  roiPct=nan
    json.dumps of that -> {"units": NaN, "roiPct": NaN, "pct": 60.0}
    node JSON.parse   -> SyntaxError: Unexpected token 'N'

The build had twelve separate `json.dumps` calls and no shared writer, so the
same accident was one arithmetic slip away in twelve places, none of them
passing `allow_nan=False`. NaN also round-trips -- Python's loader accepts the
literal it wrote -- so a bad value persists across builds rather than clearing
on the next one.

Guarded in two places on purpose. `_row_units` stops one bad row costing every
other figure in its bucket; `write_json` stops anything at all reaching a
published file. The second is the one that matters, because it does not need
to know where the value came from.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import _accumulate_summary, _summary_bucket
from shared_utils import dumps_json, json_safe, write_json

NON_FINITE = (float("nan"), float("inf"), float("-inf"))


class SanitiserTests(unittest.TestCase):
    def test_each_non_finite_value_becomes_null(self) -> None:
        for value in NON_FINITE:
            self.assertIsNone(json_safe(value))

    def test_ordinary_values_are_untouched(self) -> None:
        for value in (0.0, -1.5, 52.4, 0, -7, "text", None, True, False):
            self.assertEqual(json_safe(value), value)

    def test_it_reaches_inside_nested_structures(self) -> None:
        payload = {"a": [1.0, float("nan"), {"b": float("inf")}], "c": (float("-inf"), 2)}
        self.assertEqual(json_safe(payload), {"a": [1.0, None, {"b": None}], "c": [None, 2]})

    def test_dict_keys_and_shape_survive(self) -> None:
        cleaned = json_safe({"units": float("nan"), "pct": 60.0})
        self.assertEqual(sorted(cleaned), ["pct", "units"])
        self.assertEqual(cleaned["pct"], 60.0)


class DumperTests(unittest.TestCase):
    def test_the_bare_literals_never_appear(self) -> None:
        text = dumps_json({"units": float("nan"), "roi": float("inf"), "n": float("-inf")})
        for literal in ("NaN", "Infinity", "-Infinity"):
            self.assertNotIn(literal, text)

    def test_the_output_parses_as_strict_json(self) -> None:
        """`json.loads` accepts the literals by default, so strict parsing --
        which is what a browser does -- is the check that means anything."""
        text = dumps_json({"units": float("nan"), "pct": 60.0})
        parsed = json.loads(text, parse_constant=_reject)
        self.assertIsNone(parsed["units"])
        self.assertEqual(parsed["pct"], 60.0)

    def test_a_clean_payload_is_byte_for_byte_what_it_was(self) -> None:
        payload = {"b": 2, "a": [1, {"c": "x"}], "n": None}
        self.assertEqual(dumps_json(payload), json.dumps(payload, indent=2, default=str))


def _reject(name: str) -> None:
    raise AssertionError(f"strict JSON has no {name}")


class WriterTests(unittest.TestCase):
    def test_a_written_file_is_readable_by_a_strict_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accuracy.json"
            write_json(path, {"summary": {"units": float("nan"), "pct": 60.0}})
            parsed = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
            self.assertIsNone(parsed["summary"]["units"])

    def test_every_writer_that_feeds_a_published_file_uses_it(self) -> None:
        """A raw `json.dumps` writing into docs/data reopens the hole."""
        producers = (
            "accuracy_tracker.py", "model_fit.py", "elo.py", "espn_odds.py",
            "scripts/backtest_model.py", "scripts/backfill_history.py",
            "scripts/build_pages_data.py",
        )
        for name in producers:
            source = (ROOT / name).read_text(encoding="utf-8")
            for line_no, line in enumerate(source.split("\n"), 1):
                if "json.dumps" not in line:
                    continue
                self.assertNotIn(
                    "write_text", line,
                    f"{name}:{line_no} writes a file with a raw json.dumps: {line.strip()}",
                )

    def test_default_str_still_handles_types_json_cannot(self) -> None:
        from datetime import datetime, timezone

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            write_json(path, {"at": datetime(2026, 8, 17, tzinfo=timezone.utc)})
            self.assertIn("2026-08-17", path.read_text(encoding="utf-8"))


class BucketTests(unittest.TestCase):
    """The path the fuzzer actually walked."""

    def test_one_poisoned_row_no_longer_takes_the_bucket_with_it(self) -> None:
        bucket = _summary_bucket()
        for item in (
            {"status": "graded", "correct": True, "units": 0.9},
            {"status": "graded", "correct": True, "units": float("nan")},
            {"status": "graded", "correct": False, "units": -1.0},
        ):
            _accumulate_summary(bucket, item)
        self.assertEqual(bucket["total"], 3)
        self.assertTrue(math.isfinite(bucket["units"]), bucket["units"])
        self.assertTrue(math.isfinite(bucket["roiPct"]), bucket["roiPct"])
        # The two good rows still contribute exactly what they did before.
        self.assertAlmostEqual(bucket["units"], -0.1, places=3)

    def test_a_row_whose_units_are_not_a_number_at_all_costs_only_itself(self) -> None:
        bucket = _summary_bucket()
        for item in (
            {"status": "graded", "correct": True, "units": "not a number"},
            {"status": "graded", "correct": True, "units": 0.9},
        ):
            _accumulate_summary(bucket, item)
        self.assertEqual(bucket["total"], 2)
        self.assertAlmostEqual(bucket["units"], 0.9, places=3)

    def test_a_numeric_string_is_still_counted(self) -> None:
        """Rows written by older builds store units as JSON numbers, but a
        string that is a number is unambiguous and worth keeping."""
        bucket = _summary_bucket()
        _accumulate_summary(bucket, {"status": "graded", "correct": True, "units": "0.9"})
        self.assertAlmostEqual(bucket["units"], 0.9, places=3)


class CommittedDataTests(unittest.TestCase):
    """Whatever is on disk right now must already be parseable."""

    def test_no_published_file_carries_a_non_finite_literal(self) -> None:
        offenders = []
        for path in sorted((ROOT / "docs" / "data").glob("*.json")):
            try:
                json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
            except AssertionError as exc:
                offenders.append(f"{path.name}: {exc}")
            except json.JSONDecodeError as exc:
                offenders.append(f"{path.name}: {exc}")
        self.assertEqual(offenders, [], "published JSON a browser cannot parse")


class BrowserTests(unittest.TestCase):
    """The claim under test is about JSON.parse, so ask JSON.parse."""

    def test_node_accepts_what_the_writer_produces(self) -> None:
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("node not available")
        text = dumps_json({"summary": {"units": float("nan"), "roiPct": float("inf")}})
        script = f"const o = JSON.parse({json.dumps(text)}); if (o.summary.units !== null) throw new Error('not null');"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_node_rejects_what_it_produced_before(self) -> None:
        """Guards the premise: if JSON.parse tolerated NaN, none of this matters."""
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("node not available")
        old = json.dumps({"units": float("nan")})
        self.assertIn("NaN", old)
        script = f"JSON.parse({json.dumps(old)});"
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, "JSON.parse accepted NaN, premise is wrong")


if __name__ == "__main__":
    unittest.main()
