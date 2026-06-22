"""Load and apply data-driven calibration shrinkage from graded pick history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STRONG_THRESHOLD = 68
LEAN_THRESHOLD = 57
MIN_PICK_CONFIDENCE = LEAN_THRESHOLD
DEFAULT_SHRINK = 0.88

DEFAULT_BUCKET_SHRINK = {
    "strong_68+": 0.72,
    "lean_57+": 0.84,
    "coin_<57": 0.95,
}

CALIBRATION_FILE = "calibration.json"


def confidence_bucket(confidence_pct: float | None) -> str:
    if confidence_pct is None:
        return "coin_<57"
    if confidence_pct >= STRONG_THRESHOLD:
        return "strong_68+"
    if confidence_pct >= LEAN_THRESHOLD:
        return "lean_57+"
    return "coin_<57"


def _clamp_shrink(value: float) -> float:
    return max(0.5, min(0.98, value))


def compute_bucket_shrink(avg_predicted_pct: float, actual_win_pct: float) -> float:
    """Derive shrink multiplier from predicted vs actual win rate in a bucket."""
    predicted = avg_predicted_pct / 100.0
    actual = actual_win_pct / 100.0
    if predicted <= 0.5:
        return DEFAULT_SHRINK
    shrink = (actual - 0.5) / (predicted - 0.5)
    return _clamp_shrink(shrink)


def bucket_shrink_from_calibration_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Map confidence-range rows to tier shrink values."""
    tier_values: dict[str, list[float]] = {
        "strong_68+": [],
        "lean_57+": [],
        "coin_<57": [],
    }
    for row in rows:
        range_label = row.get("confidenceRange") or ""
        try:
            low = int(range_label.split("-")[0])
        except (TypeError, ValueError, IndexError):
            continue
        bucket = confidence_bucket(float(low))
        if row.get("picks", 0) < 3:
            continue
        tier_values[bucket].append(
            compute_bucket_shrink(
                float(row.get("avgPredictedPct") or 0.0),
                float(row.get("actualWinPct") or 0.0),
            )
        )

    result = dict(DEFAULT_BUCKET_SHRINK)
    for tier, values in tier_values.items():
        if values:
            result[tier] = _clamp_shrink(sum(values) / len(values))
    return result


def compute_calibration_params(report: dict[str, Any]) -> dict[str, Any]:
    """Build per-league bucket shrink params from a calibration report."""
    global_rows = report.get("calibration") or []
    global_shrink = bucket_shrink_from_calibration_rows(global_rows)
    by_league: dict[str, dict[str, float]] = {"default": global_shrink}

    league_rows = report.get("calibrationByLeague") or {}
    for league, rows in league_rows.items():
        if rows:
            by_league[league] = bucket_shrink_from_calibration_rows(rows)

    return {
        "defaultShrink": DEFAULT_SHRINK,
        "minPickConfidence": MIN_PICK_CONFIDENCE,
        "buckets": by_league,
        "derivedFromGraded": report.get("summary", {}).get("graded"),
    }


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def load_calibration_params(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    report = _load_json(data_dir / CALIBRATION_FILE, {})
    params = report.get("calibrationParams")
    if isinstance(params, dict) and params.get("buckets"):
        return params
    return compute_calibration_params(report)


def shrink_for_pick(
    *,
    league: str,
    confidence_pct: float | None,
    params: dict[str, Any] | None = None,
) -> float:
    params = params or load_calibration_params()
    buckets = params.get("buckets") or {}
    league_buckets = buckets.get(league) or buckets.get("default") or DEFAULT_BUCKET_SHRINK
    tier = confidence_bucket(confidence_pct)
    return float(league_buckets.get(tier) or params.get("defaultShrink") or DEFAULT_SHRINK)


def calibrate_probability(
    prob: float,
    *,
    league: str = "mlb",
    confidence_pct: float | None = None,
    params: dict[str, Any] | None = None,
) -> float:
    """Pull probabilities toward 50% using league- and bucket-specific shrinkage."""
    shrink = shrink_for_pick(league=league, confidence_pct=confidence_pct, params=params)
    centered = prob - 0.5
    return max(0.0, min(1.0, 0.5 + centered * shrink))


def is_publishable_pick(prediction: dict[str, Any] | None) -> bool:
    if not prediction or not prediction.get("predictedWinner"):
        return False
    confidence = prediction.get("confidence")
    if confidence is None:
        return False
    return float(confidence) >= MIN_PICK_CONFIDENCE
