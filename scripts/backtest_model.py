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
from shared_utils import dumps_json, write_json  # noqa: E402

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
    write_json(output_path, report)
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
        # Scored on the current model only. The all-time curve above pools every
        # model version this log has ever carried, so it describes history
        # rather than what the board is doing now.
        "reliabilityRecent": evaluation.reliability_curve(
            evaluation.recent_pairs(observations)
        ),
        "reliabilityRecentPicks": len(evaluation.recent_pairs(observations)),
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

    # Score at the ridge strength the shipped weights were fitted with, so the
    # reported number describes the model that is actually live. l2 can be a
    # plain scalar or a per-feature dict (strengthDiff and marketLogit now get
    # their own penalty) -- walk_forward_scores accepts either, so this must
    # not force it through float() and crash the moment a dict ships.
    shipped = model_fit.load_model(data_dir)
    l2 = (shipped.metadata.get("l2") if shipped else None) or 1.0

    scores = model_fit.walk_forward_scores(samples, l2=l2)
    scores["features"] = list(model_fit.ANCHORED_FEATURES)
    scores["l2"] = l2
    return scores


def write_evaluation_report(data_dir: Path) -> dict[str, Any]:
    report = build_evaluation_report(data_dir)
    write_json(data_dir / EVALUATION_FILE, report)
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

    # This block goes first because the tables below cannot answer "is the model
    # better than the market". They score `model (published)`, which is whatever
    # version was live when each game was predicted, so a model fixed last week
    # still reads as losing for as long as its own bad history dominates the
    # record. This is the live model, out of sample, on the games where a price
    # existed -- the same games, the same fold boundaries.
    fitted_head = (report.get("fittedWalkForward") or {}).get("vsMarket") or {}
    if fitted_head.get("n"):
        verdict = "ahead of" if fitted_head["edge"] > 0 else "behind"
        print("  LIVE MODEL vs MARKET, out of sample, same games")
        print(f"    model {fitted_head['modelLogLoss']} · market {fitted_head['marketLogLoss']}"
              f" · n={fitted_head['n']}")
        print(f"    the model is {verdict} the market by {abs(fitted_head['edge']):.4f} logloss\n")

    print("  All graded games -- POOLED ACROSS EVERY MODEL VERSION, read as history")
    _print_forecaster_table(report["overall"])

    print("\n  Games with market odds (same slate for every forecaster)")
    _print_forecaster_table(report["vsMarket"])

    divergence = report["divergence"]
    if divergence.get("n"):
        # Current pipeline first, because that is the model anyone can act on.
        # The pooled row is kept underneath for the trend, clearly labelled --
        # it was the only row for a while and reported June's 19.2-point median
        # as though it described the model running today.
        for label, block in (("current pipeline", divergence.get("current") or {}),
                             ("all versions pooled", divergence)):
            if not block.get("n"):
                continue
            print(f"\n  Model vs market ({label}, n={block['n']})")
            print(f"    median gap {block['medianGapPct']}pts · mean {block['meanGapPct']}pts"
                  f" · {block['shareOver15Pct']}% of games diverge >15pts")
            agree = block["agreesWithMarket"]
            fade = block["fadesMarket"]
            print(f"    agrees with market: {agree['picks']} picks, {agree['winPct']}% win")
            err = "" if fade.get("stdErrPct") is None else f" ±{fade['stdErrPct']}"
            print(f"    fades market:       {fade['picks']} picks ({fade['sharePct']}%),"
                  f" {fade['winPct']}%{err} win (break-even {fade['breakEvenPct']}%)")

    fitted = report.get("fittedWalkForward") or {}
    if fitted.get("n"):
        print("\n  Fitted model, walk-forward (out-of-sample, model currently live)")
        print(f"    features {'+'.join(fitted['features'])} · {fitted['folds']} folds · n={fitted['n']}")
        print(f"    logloss {fitted['logLoss']} · brier {fitted['brier']} · acc {fitted['accuracy']}")
        head = fitted.get("vsMarket") or {}
        if head.get("n"):
            print(f"    against the market on the {head['n']} priced games of those:"
                  f" model {head['modelLogLoss']} · market {head['marketLogLoss']}"
                  f" · edge {head['edge']:+.4f}")

    live_bias = (report.get("fittedWalkForward") or {}).get("homeBias") or {}
    if live_bias:
        print("\n  Home bias, LIVE model out of sample (this is the one to act on)")
        for league, stats in sorted(live_bias.items()):
            verdict = "SIGNIFICANT" if stats["significant"] else "within noise"
            print(f"    {league:<10} picks home {stats['pickHomePct']:>5}%"
                  f" · home wins {stats['actualHomeWinPct']:>5}%"
                  f" · bias {stats['biasPct']:+.1f} ±{stats['stdErrPct']:.1f}pts"
                  f" (n={stats['n']}) -- {verdict}")

    print("\n  Home bias, published picks pooled across model versions -- history")
    for league, stats in sorted(report["homeBias"].items()):
        print(f"    {league:<10} picks home {stats['pickHomePct']:>5}%"
              f" · home wins {stats['actualHomeWinPct']:>5}%"
              f" · bias {stats['biasPct']:+.1f}pts (n={stats['n']})")

    def _print_reliability(title: str, rows: list[dict[str, Any]]) -> None:
        print(f"\n  {title}")
        # The aggregate first, because at this sample size it is usually the only
        # conclusive number: every individual bucket can be too thin to read
        # while the pooled figure is solid. Reporting only the buckets invited
        # exactly the wrong conclusion -- a noisy +22 on n=10 looks alarming and
        # says nothing.
        picks = sum(row["picks"] for row in rows)
        if picks:
            stated = sum(row["avgPredictedPct"] * row["picks"] for row in rows) / picks
            actual = sum(row["actualWinPct"] * row["picks"] for row in rows) / picks
            verdict = "well calibrated" if abs(stated - actual) <= 3 else "miscalibrated"
            print(f"    OVERALL: predicted {stated:.1f}% · actual {actual:.1f}%"
                  f" · miss {stated - actual:+.1f}pts (n={picks}) -- {verdict}")
        for row in rows:
            flag = (
                ""
                if row["picks"] >= evaluation.MIN_BUCKET_FOR_CONCLUSION
                else "   <- too thin to conclude"
            )
            print(f"    {row['range']}%: predicted {row['avgPredictedPct']}%"
                  f" · actual {row['actualWinPct']}% ±{row['stdErrPct']}"
                  f" · miss {row['overconfidencePct']:+.1f}pts (n={row['picks']}){flag}")

    recent_n = report.get("reliabilityRecentPicks") or 0
    _print_reliability(
        f"Reliability, CURRENT model (last {recent_n} graded picks)",
        report.get("reliabilityRecent") or [],
    )
    _print_reliability(
        "Reliability, all-time -- pools every past model version, read as history",
        report["reliability"],
    )


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
            print(dumps_json(report))
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
        print(dumps_json(report))
        return 0

    report = write_calibration_report(args.data_dir) if args.write else summarize_predictions(args.data_dir)
    if args.json or args.write:
        if not args.write:
            print(dumps_json(report))
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
