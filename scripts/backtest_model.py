#!/usr/bin/env python3
"""Backtest model picks from predictions_log.json and graded accuracy data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import ACCURACY_FILE, LOG_FILE  # noqa: E402

CALIBRATION_FILE = "calibration.json"
STRONG_THRESHOLD = 68
LEAN_THRESHOLD = 57


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _bucket_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= STRONG_THRESHOLD:
        return "strong_68+"
    if confidence >= LEAN_THRESHOLD:
        return "lean_57+"
    return "coin_<57"


def _calibration_buckets(graded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"predicted": 0.0, "actual": 0.0, "count": 0})
    for item in graded:
        confidence = item.get("confidence")
        if confidence is None:
            continue
        bucket = int(min(90, max(50, round(confidence / 5) * 5)))
        key = f"{bucket}-{bucket + 4}"
        buckets[key]["predicted"] += confidence / 100.0
        buckets[key]["actual"] += 1.0 if item.get("correct") else 0.0
        buckets[key]["count"] += 1

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets, key=lambda value: int(value.split("-")[0])):
        bucket = buckets[key]
        count = bucket["count"]
        if not count:
            continue
        avg_predicted = round(bucket["predicted"] / count * 100, 1)
        actual_win = round(bucket["actual"] / count * 100, 1)
        rows.append(
            {
                "confidenceRange": key,
                "picks": count,
                "avgPredictedPct": avg_predicted,
                "actualWinPct": actual_win,
                "overconfidencePct": round(avg_predicted - actual_win, 1),
            }
        )
    return rows


def _coverage_breakdown(graded: list[dict[str, Any]]) -> dict[str, Any]:
    flags = ("restData", "scheduleFlags", "impliedOdds", "lineup", "espnPredictor")
    breakdown: dict[str, dict[str, Any]] = {}
    for record in graded:
        coverage = (record.get("features") or {}).get("dataCoverage") or {}
        for flag in flags:
            bucket = breakdown.setdefault(flag, {"with": 0, "without": 0, "winsWith": 0, "winsWithout": 0})
            has_flag = bool(coverage.get(flag))
            if has_flag:
                bucket["with"] += 1
                if record.get("correct"):
                    bucket["winsWith"] += 1
            else:
                bucket["without"] += 1
                if record.get("correct"):
                    bucket["winsWithout"] += 1
    return breakdown


def summarize_predictions(data_dir: Path) -> dict[str, Any]:
    log = _load_json(data_dir / LOG_FILE, {"predictions": {}})
    accuracy = _load_json(data_dir / ACCURACY_FILE, {"picksByEventId": {}})
    picks_by_event = accuracy.get("picksByEventId") or {}

    graded: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for event_id, pending_pick in (log.get("predictions") or {}).items():
        record = picks_by_event.get(event_id) or {"status": "pending", **pending_pick}
        merged = {**pending_pick, **record, "eventId": event_id}
        if merged.get("status") == "graded":
            graded.append(merged)
        else:
            pending.append(merged)

    by_league: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "units": 0.0, "pending": 0}
    )
    by_confidence: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "units": 0.0}
    )

    for item in graded:
        league = item.get("league") or "unknown"
        bucket = _bucket_confidence(item.get("confidence"))
        for target in (by_league[league], by_confidence[bucket]):
            target["total"] += 1
            if item.get("correct"):
                target["correct"] += 1
            target["units"] = round(target.get("units", 0.0) + float(item.get("units") or 0.0), 3)
        if by_league[league]["total"]:
            by_league[league]["winPct"] = round(by_league[league]["correct"] / by_league[league]["total"] * 100, 1)
            by_league[league]["roiPct"] = round(by_league[league]["units"] / by_league[league]["total"] * 100, 1)
        if by_confidence[bucket]["total"]:
            by_confidence[bucket]["winPct"] = round(
                by_confidence[bucket]["correct"] / by_confidence[bucket]["total"] * 100,
                1,
            )

    for item in pending:
        league = item.get("league") or "unknown"
        by_league[league]["pending"] = by_league[league].get("pending", 0) + 1

    total_graded = len(graded)
    total_correct = sum(1 for item in graded if item.get("correct"))
    total_units = round(sum(float(item.get("units") or 0.0) for item in graded), 3)

    feature_coverage = {
        "withFeatures": sum(1 for item in log.get("predictions", {}).values() if item.get("features")),
        "totalLogged": len(log.get("predictions") or {}),
        "withRestData": sum(
            1
            for item in log.get("predictions", {}).values()
            if (item.get("features") or {}).get("dataCoverage", {}).get("restData")
        ),
    }

    calibration = _calibration_buckets(graded)
    avg_overconfidence = None
    if calibration:
        avg_overconfidence = round(
            sum(row["overconfidencePct"] for row in calibration) / len(calibration),
            1,
        )

    return {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "thresholds": {"strong": STRONG_THRESHOLD, "lean": LEAN_THRESHOLD},
        "summary": {
            "graded": total_graded,
            "correct": total_correct,
            "winPct": round(total_correct / total_graded * 100, 1) if total_graded else None,
            "units": total_units,
            "roiPct": round(total_units / total_graded * 100, 1) if total_graded else None,
            "pending": len(pending),
            "featureCoverage": feature_coverage,
            "avgOverconfidencePct": avg_overconfidence,
        },
        "byLeague": dict(by_league),
        "byConfidence": dict(by_confidence),
        "calibration": calibration,
        "coverageBreakdown": _coverage_breakdown(graded),
    }


def write_calibration_report(data_dir: Path) -> dict[str, Any]:
    report = summarize_predictions(data_dir)
    output_path = data_dir / CALIBRATION_FILE
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest logged model picks.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "docs" / "data",
        help="Directory containing predictions_log.json and accuracy.json",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument("--write", action="store_true", help=f"Write {CALIBRATION_FILE} to data dir")
    args = parser.parse_args()

    report = write_calibration_report(args.data_dir) if args.write else summarize_predictions(args.data_dir)
    if args.json or args.write:
        if not args.write:
            print(json.dumps(report, indent=2))
        else:
            print(f"Wrote {args.data_dir / CALIBRATION_FILE}")
        return 0

    summary = report["summary"]
    print("Model backtest")
    print(f"  Graded picks: {summary['graded']}")
    print(f"  Win rate:     {summary['winPct']}%")
    print(f"  Units:        {summary['units']}")
    print(f"  ROI:          {summary['roiPct']}%")
    print(f"  Pending:      {summary['pending']}")
    coverage = summary["featureCoverage"]
    print(f"  Feature log:  {coverage['withFeatures']}/{coverage['totalLogged']}")
    print(f"  Rest data:    {coverage.get('withRestData', 0)}/{coverage['totalLogged']}")

    print("\nBy league")
    for league, stats in sorted(report["byLeague"].items()):
        print(
            f"  {league}: {stats.get('correct', 0)}-{stats.get('total', 0) - stats.get('correct', 0)}"
            f" ({stats.get('winPct', '—')}%) · ROI {stats.get('roiPct', '—')}% · pending {stats.get('pending', 0)}"
        )

    if report["calibration"]:
        print("\nCalibration (confidence bucket vs actual win%)")
        for row in report["calibration"]:
            print(
                f"  {row['confidenceRange']}%: predicted {row['avgPredictedPct']}%"
                f" · actual {row['actualWinPct']}% · n={row['picks']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
