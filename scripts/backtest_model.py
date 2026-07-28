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
from calibration_params import compute_calibration_params, compute_platt_params  # noqa: E402
from mlb_predictions import apply_predictions  # noqa: E402
import model_fit  # noqa: E402
from scripts import evaluation  # noqa: E402

CALIBRATION_FILE = "calibration.json"
EVALUATION_FILE = "evaluation.json"
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


def _calibration_by_league(graded: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in graded:
        league = item.get("league") or "unknown"
        grouped[league].append(item)
    return {league: _calibration_buckets(items) for league, items in grouped.items()}


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
    calibration_by_league = _calibration_by_league(graded)
    avg_overconfidence = None
    if calibration:
        avg_overconfidence = round(
            sum(row["overconfidencePct"] for row in calibration) / len(calibration),
            1,
        )

    report = {
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
        "calibrationByLeague": calibration_by_league,
        "coverageBreakdown": _coverage_breakdown(graded),
    }
    report["calibrationParams"] = compute_calibration_params(report)
    # Fitted on raw, pre-calibration confidence and keyed by probability method,
    # so the curve never learns from its own previous corrections.
    report["plattParams"] = compute_platt_params(graded)
    return report


def write_calibration_report(data_dir: Path) -> dict[str, Any]:
    report = summarize_predictions(data_dir)
    output_path = data_dir / CALIBRATION_FILE
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_evaluation_report(data_dir: Path) -> dict[str, Any]:
    """Score the model's probabilities against the baselines it must beat.

    Win rate hides whether the probabilities mean anything, so this reports
    log loss / Brier / AUC alongside naive forecasters and the market.
    """
    observations = evaluation.load_observations(data_dir)

    by_league: dict[str, Any] = {}
    for league in sorted({item.league for item in observations}):
        pool = [item for item in observations if item.league == league]
        by_league[league] = {
            "overall": evaluation.compare_forecasters(pool),
            "vsMarket": evaluation.compare_forecasters(pool, require_market=True),
            "divergence": evaluation.divergence_report(pool),
        }

    published = [(item.published, item.home_won) for item in observations if item.published is not None]

    return {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "n": len(observations),
        "overall": evaluation.compare_forecasters(observations),
        # Restricted to games with odds so every forecaster faces the same slate.
        "vsMarket": evaluation.compare_forecasters(observations, require_market=True),
        "divergence": evaluation.divergence_report(observations),
        "homeBias": evaluation.home_bias_report(observations),
        "reliability": evaluation.reliability_curve(published),
        "byLeague": by_league,
        "fittedWalkForward": _fitted_walk_forward(data_dir),
    }


def _fitted_walk_forward(data_dir: Path) -> dict[str, Any]:
    """Out-of-sample score for the fitted model.

    The published numbers above come from whatever model was live when each game
    was predicted. This is the honest score for the model running now: trained
    only on games that preceded each test fold, so it never sees its own answers.
    """
    try:
        samples, _ = model_fit.samples_from_log(data_dir)
    except Exception:  # pragma: no cover - defensive around malformed logs
        return {}
    if not samples:
        return {}
    scores = model_fit.walk_forward_scores(samples)
    scores["features"] = list(model_fit.ANCHORED_FEATURES)
    return scores


def write_evaluation_report(data_dir: Path) -> dict[str, Any]:
    report = build_evaluation_report(data_dir)
    (data_dir / EVALUATION_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_forecaster_table(block: dict[str, Any]) -> None:
    if not block.get("n"):
        print("    (no observations)")
        return
    print(f"    n={block['n']}  home base rate {block['homeBaseRate'] * 100:.1f}%")
    print(f"    {'forecaster':<42} {'logloss':>8} {'brier':>8} {'auc':>7} {'acc':>7}")
    for row in block["forecasters"]:
        def fmt(value: Any, width: int, digits: int = 4) -> str:
            return f"{value:>{width}.{digits}f}" if value is not None else f"{'—':>{width}}"

        flag = "" if row.get("reliable") else "  (small n)"
        print(
            f"    {row['name']:<42} {fmt(row['logLoss'], 8)} {fmt(row['brier'], 8)}"
            f" {fmt(row['auc'], 7)} {fmt(row['accuracy'], 7)}{flag}"
        )


def print_evaluation(report: dict[str, Any]) -> None:
    print("Model evaluation")
    print(f"  Graded observations: {report['n']}\n")

    print("  All graded games")
    _print_forecaster_table(report["overall"])

    print("\n  Games with market odds (same slate for every forecaster)")
    _print_forecaster_table(report["vsMarket"])

    divergence = report["divergence"]
    if divergence.get("n"):
        print("\n  Model vs market")
        print(f"    median gap {divergence['medianGapPct']}pts · mean {divergence['meanGapPct']}pts"
              f" · {divergence['shareOver15Pct']}% of games diverge >15pts")
        agree = divergence["agreesWithMarket"]
        fade = divergence["fadesMarket"]
        print(f"    agrees with market: {agree['picks']} picks, {agree['winPct']}% win")
        print(f"    fades market:       {fade['picks']} picks, {fade['winPct']}% win"
              f" (break-even {fade['breakEvenPct']}%)")

    fitted = report.get("fittedWalkForward") or {}
    if fitted.get("n"):
        print("\n  Fitted model, walk-forward (out-of-sample, model currently live)")
        print(f"    features {'+'.join(fitted['features'])} · {fitted['folds']} folds · n={fitted['n']}")
        print(f"    logloss {fitted['logLoss']} · brier {fitted['brier']} · acc {fitted['accuracy']}")

    print("\n  Home bias (pick rate vs actual home win rate)")
    for league, stats in sorted(report["homeBias"].items()):
        print(f"    {league:<10} picks home {stats['pickHomePct']:>5}%"
              f" · home wins {stats['actualHomeWinPct']:>5}%"
              f" · bias {stats['biasPct']:+.1f}pts (n={stats['n']})")

    print("\n  Reliability (confidence bucket vs actual)")
    for row in report["reliability"]:
        print(f"    {row['range']}%: predicted {row['avgPredictedPct']}%"
              f" · actual {row['actualWinPct']}% ±{row['stdErrPct']}"
              f" · miss {row['overconfidencePct']:+.1f}pts (n={row['picks']})")


def _actual_winner_from_snapshot_game(game: dict[str, Any]) -> str | None:
    if not game.get("isFinal"):
        return None
    home_score = game.get("homeScore")
    away_score = game.get("awayScore")
    if home_score is None or away_score is None:
        return None
    try:
        home = int(home_score)
        away = int(away_score)
    except (TypeError, ValueError):
        return None
    if home == away:
        return "Draw"
    return game.get("homeTeam") if home > away else game.get("awayTeam")


def replay_snapshot(
    data_dir: Path,
    *,
    league: str,
    schedule_date: str,
) -> dict[str, Any]:
    """Replay a dated snapshot through predict_game and compare to known finals."""
    snapshot_path = data_dir / f"{league}_{schedule_date}.json"
    payload = _load_json(snapshot_path, {})
    games = payload.get("games") or []
    replayed = apply_predictions([dict(game) for game in games])

    results: list[dict[str, Any]] = []
    for game in replayed:
        prediction = game.get("prediction") or {}
        actual = _actual_winner_from_snapshot_game(game)
        if not prediction.get("predictedWinner") or actual is None:
            continue
        predicted = prediction.get("predictedWinner")
        correct = predicted == actual or (
            predicted != "Draw"
            and actual != "Draw"
            and predicted in (game.get("homeTeam"), game.get("awayTeam"))
            and actual in (game.get("homeTeam"), game.get("awayTeam"))
            and predicted == actual
        )
        results.append(
            {
                "eventId": game.get("eventId"),
                "matchup": game.get("matchup"),
                "predicted": predicted,
                "actual": actual,
                "correct": correct,
                "confidence": prediction.get("confidence"),
            }
        )

    correct = sum(1 for item in results if item.get("correct"))
    total = len(results)
    return {
        "league": league,
        "scheduleDate": schedule_date,
        "snapshotPath": str(snapshot_path),
        "gamesReplayed": len(replayed),
        "finalsCompared": total,
        "correct": correct,
        "winPct": round(correct / total * 100, 1) if total else None,
        "results": results,
    }


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
    parser.add_argument("--replay", action="store_true", help="Replay a dated snapshot JSON through the model")
    parser.add_argument("--league", default="mlb", help="League id for snapshot replay")
    parser.add_argument("--date", dest="schedule_date", help="Schedule date (YYYY-MM-DD) for snapshot replay")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Score model probabilities (log loss/Brier/AUC) against baselines and the market",
    )
    args = parser.parse_args()

    if args.evaluate:
        report = write_evaluation_report(args.data_dir) if args.write else build_evaluation_report(args.data_dir)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_evaluation(report)
            if args.write:
                print(f"\nWrote {args.data_dir / EVALUATION_FILE}")
        return 0

    if args.replay:
        if not args.schedule_date:
            print("Snapshot replay requires --date YYYY-MM-DD", file=sys.stderr)
            return 2
        report = replay_snapshot(args.data_dir, league=args.league, schedule_date=args.schedule_date)
        print(json.dumps(report, indent=2))
        return 0

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

    params = report.get("calibrationParams") or {}
    buckets = params.get("buckets") or {}
    if buckets:
        print("\nCalibration shrink multipliers")
        for league, values in sorted(buckets.items()):
            print(f"  {league}: {values}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
