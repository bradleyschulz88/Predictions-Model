"""Probability calibration fitted from graded pick history.

Two mechanisms live here.

**Platt scaling** (preferred) maps a raw probability through
``sigmoid(a * logit(p) + b)`` with ``a`` and ``b`` fitted by maximum likelihood.
It is the right tool at this sample size, and unlike the bucket shrinkage below
it can express "this model is worse than a coin flip" as a negative slope
instead of clamping and losing the information.

**Bucket shrinkage** (legacy) pulls probabilities toward 50% by a per-tier
multiplier. It is retained only for the heuristic fallback path.

Both are fitted on *raw*, pre-calibration confidence. Fitting on published
confidence -- which a previous calibration already shrank -- compounds every
build, and it drove the MLB multipliers down to their 0.5 floor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from model_fit import fit_logistic, logit, sigmoid

STRONG_THRESHOLD = 68
LEAN_THRESHOLD = 57

# Minimum published confidence for a pick. Retuned against the fitted model's
# honest distribution -- see scripts/backtest_model.py --evaluate. Kept in sync
# with MIN_PUBLISHABLE_CONFIDENCE in dashboard/app.js.
MIN_PICK_CONFIDENCE = 55

# Per-league overrides of the publish bar. EMPTY ON PURPOSE -- read this before
# adding one, because the obvious version of this idea is wrong.
#
# MLB carried a 65 bar here briefly. The evidence looked strong: 164 priced
# graded picks in the 55-65 band hit 45.1% into prices implying 58-62%, for
# -16.4% ROI, stable across both halves of the history and both home and away.
#
# It was still wrong, because a fixed cutoff was pinned to a distribution that
# then moved underneath it. MLB's median stated confidence ran 65-73 through
# 2026-07-23 and 54-61 from 2026-07-24, a 9.6-point drop at the median, when
# Platt calibration started correcting the model's overconfidence. The bar was
# measured where it excluded the bottom quartile and ended up excluding the
# middle: it withheld 90% of MLB games, and 100% on several days, leaving an
# empty board.
#
# The band was never the problem. Overconfidence was, and calibration fixed it:
#
#     graded MLB      n     mean stated   actual   gap
#     before 07-24   432       67.6%      55.1%   +12.5 pts
#     after  07-24    71       59.8%      59.2%    +0.6 pts
#
# A stated 60 used to mean roughly 45, which loses money at any price. It now
# means roughly 60, which clears the 52.4% break-even at -110. Withholding that
# band today would be discarding the picks calibration just repaired.
#
# So if a league ever looks like it needs its own bar, first check whether the
# distribution has moved -- the two are very hard to tell apart from the band
# statistics alone. If a bar is genuinely needed, derive it from the CURRENT
# distribution and re-derive it whenever the model is refit.
MIN_PICK_CONFIDENCE_BY_LEAGUE: dict[str, float] = {}


def min_pick_confidence(league: str | None = None) -> float:
    """Publish threshold for a league, falling back to the global minimum."""
    if league:
        override = MIN_PICK_CONFIDENCE_BY_LEAGUE.get(str(league).lower())
        if override is not None:
            return float(override)
    return float(MIN_PICK_CONFIDENCE)


DEFAULT_SHRINK = 0.88

DEFAULT_BUCKET_SHRINK = {
    "strong_68+": 0.72,
    "lean_57+": 0.84,
    "coin_<57": 0.95,
}

CALIBRATION_FILE = "calibration.json"

# A calibration curve needs this many graded picks before it is trusted at all,
# and this many again before it fully displaces the identity mapping.
MIN_PLATT_SAMPLES = 40
PLATT_SHRINK_PRIOR = 120.0

IDENTITY_PLATT = {"a": 1.0, "b": 0.0, "n": 0}


def confidence_bucket(confidence_pct: float | None) -> str:
    if confidence_pct is None:
        return "coin_<57"
    if confidence_pct >= STRONG_THRESHOLD:
        return "strong_68+"
    if confidence_pct >= LEAN_THRESHOLD:
        return "lean_57+"
    return "coin_<57"


# --------------------------------------------------------------------------
# Platt scaling
# --------------------------------------------------------------------------


def fit_platt(pairs: list[tuple[float, int]]) -> dict[str, Any]:
    """Fit sigmoid(a * logit(p) + b), shrunk toward identity on thin samples.

    Platt rather than isotonic: with a few hundred picks spread across buckets
    that can hold a single game, a two-parameter fit is the most the data
    supports without chasing noise.
    """
    usable = [(prob, outcome) for prob, outcome in pairs if prob is not None]
    if len(usable) < MIN_PLATT_SAMPLES:
        return dict(IDENTITY_PLATT, n=len(usable))

    rows = [[1.0, logit(prob)] for prob, _ in usable]
    labels = [outcome for _, outcome in usable]
    weights = fit_logistic(rows, labels, l2=1e-3)
    if len(weights) < 2 or not all(math.isfinite(value) for value in weights):
        return dict(IDENTITY_PLATT, n=len(usable))

    intercept, slope = weights[0], weights[1]

    # Shrink toward the identity mapping (a=1, b=0) so a short history nudges
    # the curve rather than replacing it.
    weight = len(usable) / (len(usable) + PLATT_SHRINK_PRIOR)
    return {
        "a": round(1.0 + (slope - 1.0) * weight, 4),
        "b": round(intercept * weight, 4),
        "n": len(usable),
    }


def apply_platt(prob: float, params: dict[str, Any] | None) -> float:
    """Map a raw probability through a fitted calibration curve."""
    if not params:
        return prob
    slope = float(params.get("a", 1.0))
    intercept = float(params.get("b", 0.0))
    return sigmoid(slope * logit(prob) + intercept)


def platt_params_for(
    calibration: dict[str, Any] | None,
    *,
    method: str,
    league: str,
) -> dict[str, Any]:
    """Look up the curve for a league, falling back to pooled then identity.

    Keyed by method so a curve fitted against the heuristic's errors is never
    applied to the fitted model, which makes different mistakes.
    """
    by_method = ((calibration or {}).get("plattParams") or {}).get(method) or {}
    for key in (league, "default"):
        params = by_method.get(key)
        if params and params.get("n", 0) >= MIN_PLATT_SAMPLES:
            return params
    return dict(IDENTITY_PLATT)


def _home_outcome(record: dict[str, Any]) -> int | None:
    """1 if the home side won, 0 if the away side did, None for draws."""
    home_score, away_score = record.get("homeScore"), record.get("awayScore")
    if home_score is None or away_score is None:
        return None
    try:
        home, away = int(home_score), int(away_score)
    except (TypeError, ValueError):
        return None
    if home == away:
        return None
    return 1 if home > away else 0


def compute_platt_params(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit one curve per (probability method, league), plus a pooled default.

    Fitted in *home*-probability space rather than on folded pick-side
    confidence: a curve learned only from values above 0.5 is not valid below
    it, and applying one would bias every away pick.

    Records without a raw home probability are skipped rather than falling back
    to the published value -- that is exactly the feedback loop this replaces.
    """
    by_method: dict[str, dict[str, list[tuple[float, int]]]] = {}

    for record in graded:
        raw = record.get("rawHomeWinPct")
        if raw is None:
            raw = (record.get("features") or {}).get("rawHomeWinPct")
        if raw is None:
            continue
        outcome = _home_outcome(record)
        if outcome is None:
            continue

        method = (
            record.get("probabilityMethod")
            or (record.get("features") or {}).get("probabilityMethod")
            or "heuristic"
        )
        league = record.get("league") or "unknown"
        pair = (min(0.999, max(0.001, float(raw) / 100.0)), outcome)
        by_method.setdefault(method, {}).setdefault(league, []).append(pair)
        by_method[method].setdefault("default", []).append(pair)

    return {
        method: {league: fit_platt(pairs) for league, pairs in leagues.items()}
        for method, leagues in by_method.items()
    }


# --------------------------------------------------------------------------
# Legacy bucket shrinkage (heuristic fallback only)
# --------------------------------------------------------------------------


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
        "minPickConfidenceByLeague": dict(MIN_PICK_CONFIDENCE_BY_LEAGUE),
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


def _calibration_report(data_dir: Path | None = None) -> dict[str, Any]:
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    return _load_json(data_dir / CALIBRATION_FILE, {})


def load_calibration_params(data_dir: Path | None = None) -> dict[str, Any]:
    report = _calibration_report(data_dir)
    params = report.get("calibrationParams")
    if isinstance(params, dict) and params.get("buckets"):
        return params
    return compute_calibration_params(report)


def load_platt_params(data_dir: Path | None = None) -> dict[str, Any]:
    return _calibration_report(data_dir).get("plattParams") or {}


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


def is_publishable_pick(
    prediction: dict[str, Any] | None, league: str | None = None
) -> bool:
    """Whether a pick clears its league's publish threshold.

    The league is read from the prediction's own features when not passed, so
    existing single-argument callers keep working and still get the per-league
    bar.
    """
    if not prediction or not prediction.get("predictedWinner"):
        return False
    confidence = prediction.get("confidence")
    if confidence is None:
        return False
    if league is None:
        league = (prediction.get("features") or {}).get("league")
    return float(confidence) >= min_pick_confidence(league)
