"""Scoring metrics and baselines for the prediction model (stdlib only).

Accuracy alone is a poor guide for a probability model: a forecaster that always
says "60% home" can look respectable on win rate while carrying no information.
These metrics score the *probabilities*, and every model number is reported next
to naive baselines so that "better than nothing" is an explicit test rather than
an assumption.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

EPSILON = 1e-9
# Below this many observations a metric is noise; report it but flag it.
MIN_RELIABLE_SAMPLE = 30


def _clip(prob: float) -> float:
    return max(EPSILON, min(1.0 - EPSILON, prob))


def log_loss(pairs: Sequence[tuple[float, int]]) -> float | None:
    """Mean negative log likelihood. Lower is better; 0.693 == coin flip."""
    if not pairs:
        return None
    total = 0.0
    for prob, outcome in pairs:
        p = _clip(prob)
        total -= math.log(p) if outcome else math.log(1.0 - p)
    return total / len(pairs)


def brier_score(pairs: Sequence[tuple[float, int]]) -> float | None:
    """Mean squared error of the probability. Lower is better; 0.25 == always 50%."""
    if not pairs:
        return None
    return sum((prob - outcome) ** 2 for prob, outcome in pairs) / len(pairs)


def auc(pairs: Sequence[tuple[float, int]]) -> float | None:
    """Rank-based ROC AUC via the Mann-Whitney U identity, with tie correction.

    0.5 means the probabilities carry no ordering information at all, which is
    the failure mode this harness exists to catch.
    """
    positives = [prob for prob, outcome in pairs if outcome]
    negatives = [prob for prob, outcome in pairs if not outcome]
    if not positives or not negatives:
        return None

    ordered = sorted(pairs, key=lambda item: item[0])
    ranks: list[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        # Average rank across the tied block so ties score 0.5, not 0 or 1.
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1

    positive_rank_sum = sum(
        rank for rank, (_, outcome) in zip(ranks, ordered) if outcome
    )
    n_pos = len(positives)
    n_neg = len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy(pairs: Sequence[tuple[float, int]]) -> float | None:
    """Share of games where the higher-probability side actually won."""
    if not pairs:
        return None
    hits = sum(1 for prob, outcome in pairs if (prob >= 0.5) == bool(outcome))
    return hits / len(pairs)


# Reliability is the one metric that must not be pooled across model versions.
# A refit changes what a stated 60% means, so mixing six weeks of an
# overconfident model with a week of a corrected one reports the old model's
# error as if it were current -- which it did: the all-time table showed the
# 55-65% band running +11 points hot while the live model was running -1.8.
# Windowed by TIME, not by count. A fixed pick count sounds equivalent and is
# not: 250 picks reaches back months here, so it re-pools the model versions the
# window exists to separate.
RECENT_RELIABILITY_DAYS = 14

# Below this a bucket's binomial error swamps the effect being measured -- at
# n=17 the standard error is about 12 points, which is larger than any
# miscalibration worth acting on. Such buckets are reported and flagged rather
# than hidden, because "we cannot tell yet" is the finding.
MIN_BUCKET_FOR_CONCLUSION = 30


def recent_pairs(
    observations: Sequence[Any], *, days: int = RECENT_RELIABILITY_DAYS
) -> list[tuple[float, int]]:
    """Graded picks from the last `days`, for scoring the model as it stands now.

    Reliability is the one metric that must never be pooled across model
    versions: a refit changes what a stated 60% means. Mixing six weeks of an
    overconfident model with a week of a corrected one reports the old model's
    error as current, which is exactly what the all-time table did.
    """
    from datetime import date, timedelta

    dated = [item for item in observations if item.published is not None and item.date]
    if not dated:
        return []
    latest = max(item.date for item in dated)
    try:
        cutoff = (date.fromisoformat(latest) - timedelta(days=days)).isoformat()
    except ValueError:
        return [(item.published, item.home_won) for item in dated]
    return [(item.published, item.home_won) for item in dated if item.date >= cutoff]


def reliability_curve(
    pairs: Sequence[tuple[float, int]],
    *,
    bin_width: float = 0.05,
) -> list[dict[str, Any]]:
    """Predicted vs actual rate per probability bucket, folded onto the pick side.

    Probabilities are folded to [0.5, 1.0] so that "80% home" and "80% away" land
    in the same bucket -- that is the number the dashboard shows as confidence.
    """
    buckets: dict[int, dict[str, float]] = {}
    for prob, outcome in pairs:
        # Fold: score the side the model actually picked.
        picked_home = prob >= 0.5
        confidence = prob if picked_home else 1.0 - prob
        hit = 1.0 if bool(outcome) == picked_home else 0.0
        key = min(int(confidence / bin_width), int(1.0 / bin_width) - 1)
        bucket = buckets.setdefault(key, {"predicted": 0.0, "actual": 0.0, "count": 0.0})
        bucket["predicted"] += confidence
        bucket["actual"] += hit
        bucket["count"] += 1

    rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        count = int(bucket["count"])
        predicted_pct = bucket["predicted"] / count * 100
        actual_pct = bucket["actual"] / count * 100
        rows.append(
            {
                "range": f"{key * bin_width * 100:.0f}-{(key + 1) * bin_width * 100:.0f}",
                "picks": count,
                "avgPredictedPct": round(predicted_pct, 1),
                "actualWinPct": round(actual_pct, 1),
                "overconfidencePct": round(predicted_pct - actual_pct, 1),
                # Binomial standard error, so a gap can be judged against noise.
                "stdErrPct": round(
                    math.sqrt(max(actual_pct, EPSILON) * (100 - actual_pct) / count), 1
                ),
            }
        )
    return rows


def score(name: str, pairs: Sequence[tuple[float, int]]) -> dict[str, Any]:
    """Full metric bundle for one forecaster."""
    return {
        "name": name,
        "n": len(pairs),
        "logLoss": _round(log_loss(pairs), 4),
        "brier": _round(brier_score(pairs), 4),
        "auc": _round(auc(pairs), 4),
        "accuracy": _round(accuracy(pairs), 4),
        "reliable": len(pairs) >= MIN_RELIABLE_SAMPLE,
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


# --------------------------------------------------------------------------
# Observation loading
# --------------------------------------------------------------------------


class Observation:
    """One graded game with every forecaster's home-win probability attached."""

    __slots__ = ("event_id", "league", "date", "home_won", "model", "market", "published")

    def __init__(
        self,
        *,
        event_id: str,
        league: str,
        date: str,
        home_won: int,
        model: float | None,
        market: float | None,
        published: float | None,
    ) -> None:
        self.event_id = event_id
        self.league = league
        self.date = date
        self.home_won = home_won
        self.model = model
        self.market = market
        self.published = published


def _home_won(record: dict[str, Any]) -> int | None:
    """Resolve the actual outcome, preferring scores over the derived flag."""
    home_score = record.get("homeScore")
    away_score = record.get("awayScore")
    if home_score is not None and away_score is not None:
        try:
            home, away = int(home_score), int(away_score)
        except (TypeError, ValueError):
            home = away = None
        if home is not None and home != away:
            return 1 if home > away else 0
        if home is not None:
            return None  # Draw: not a binary home/away outcome.

    # Fall back to inverting the recorded correctness against the picked side.
    side = record.get("predictedSide")
    correct = record.get("correct")
    if correct is None or side not in {"home", "away"}:
        return None
    if side == "home":
        return 1 if correct else 0
    return 0 if correct else 1


def _published_home_prob(record: dict[str, Any]) -> float | None:
    confidence = record.get("confidence")
    side = record.get("predictedSide")
    if confidence is None or side not in {"home", "away"}:
        return None
    value = float(confidence) / 100.0
    return value if side == "home" else 1.0 - value


def load_observations(data_dir: Path) -> list[Observation]:
    """Join predictions_log.json (features) with accuracy.json (outcomes)."""
    log = _load_json(data_dir / "predictions_log.json", {"predictions": {}})
    accuracy_data = _load_json(data_dir / "accuracy.json", {"picksByEventId": {}})
    picks = accuracy_data.get("picksByEventId") or {}

    observations: list[Observation] = []
    for event_id, graded in picks.items():
        if graded.get("status") != "graded":
            continue
        logged = (log.get("predictions") or {}).get(event_id) or {}
        merged = {**logged, **graded}
        home_won = _home_won(merged)
        if home_won is None:
            continue

        features = merged.get("features") or {}
        model_home = features.get("trueHome")
        implied_home = features.get("impliedHome")
        observations.append(
            Observation(
                event_id=event_id,
                league=merged.get("league") or "unknown",
                date=merged.get("date") or merged.get("scheduleDate") or "",
                home_won=home_won,
                model=float(model_home) / 100.0 if model_home is not None else None,
                market=float(implied_home) / 100.0 if implied_home is not None else None,
                published=_published_home_prob(merged),
            )
        )
    return observations


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def home_base_rate(observations: Iterable[Observation]) -> float:
    items = list(observations)
    if not items:
        return 0.5
    return sum(item.home_won for item in items) / len(items)


def compare_forecasters(
    observations: Sequence[Observation],
    *,
    require_market: bool = False,
) -> dict[str, Any]:
    """Score the model against the baselines it has to beat to be worth running.

    When require_market is set, every forecaster is scored on the *same* subset of
    games that have odds -- otherwise the market looks artificially good or bad
    purely from facing a different slate.
    """
    pool = [item for item in observations if item.market is not None] if require_market else list(observations)
    if not pool:
        return {"n": 0, "forecasters": []}

    base_rate = home_base_rate(pool)

    forecasters: list[dict[str, Any]] = []

    published = [(item.published, item.home_won) for item in pool if item.published is not None]
    forecasters.append(score("model (published)", published))

    model = [(item.model, item.home_won) for item in pool if item.model is not None]
    forecasters.append(score("model (pre-calibration)", model))

    market = [(item.market, item.home_won) for item in pool if item.market is not None]
    if market:
        forecasters.append(score("market (de-vigged)", market))

    forecasters.append(
        score("baseline: constant home base rate", [(base_rate, item.home_won) for item in pool])
    )
    forecasters.append(
        score("baseline: always 50%", [(0.5, item.home_won) for item in pool])
    )
    # Always-home as a *probability* forecaster is degenerate (infinite log loss),
    # so express it at the empirical rate and report its accuracy separately.
    forecasters.append(
        {
            "name": "baseline: always pick home (accuracy only)",
            "n": len(pool),
            "logLoss": None,
            "brier": None,
            "auc": None,
            "accuracy": round(base_rate, 4),
            "reliable": len(pool) >= MIN_RELIABLE_SAMPLE,
        }
    )

    return {
        "n": len(pool),
        "homeBaseRate": round(base_rate, 4),
        "marketCoverage": round(len(market) / len(pool), 4) if pool else 0.0,
        "forecasters": forecasters,
    }


def divergence_report(observations: Sequence[Observation]) -> dict[str, Any]:
    """How far the model strays from the market, and whether straying pays.

    Fading an efficient market is the expensive way to be wrong, so this is
    tracked as a first-class metric rather than left implicit in the win rate.
    """
    paired = [item for item in observations if item.market is not None and item.model is not None]
    if not paired:
        return {"n": 0}

    gaps = sorted(abs(item.model - item.market) * 100 for item in paired)
    median_gap = gaps[len(gaps) // 2]

    agree_hits = agree_total = fade_hits = fade_total = 0
    for item in paired:
        model_home = item.model >= 0.5
        market_home = item.market >= 0.5
        hit = 1 if bool(item.home_won) == model_home else 0
        if model_home == market_home:
            agree_total += 1
            agree_hits += hit
        else:
            fade_total += 1
            fade_hits += hit

    return {
        "n": len(paired),
        "medianGapPct": round(median_gap, 1),
        "meanGapPct": round(sum(gaps) / len(gaps), 1),
        "shareOver15Pct": round(sum(1 for gap in gaps if gap > 15) / len(gaps) * 100, 1),
        "agreesWithMarket": {
            "picks": agree_total,
            "winPct": round(agree_hits / agree_total * 100, 1) if agree_total else None,
        },
        "fadesMarket": {
            "picks": fade_total,
            "winPct": round(fade_hits / fade_total * 100, 1) if fade_total else None,
            # -110 juice needs 52.38% to break even; below that, fading burns money.
            "breakEvenPct": 52.4,
        },
    }


def home_bias_report(observations: Sequence[Observation]) -> dict[str, Any]:
    """Does the model pick home more often than home actually wins?"""
    by_league: dict[str, dict[str, Any]] = {}
    for league in sorted({item.league for item in observations} | {"ALL"}):
        pool = observations if league == "ALL" else [i for i in observations if i.league == league]
        picks = [item for item in pool if item.published is not None]
        if not picks:
            continue
        picked_home = sum(1 for item in picks if item.published >= 0.5) / len(picks)
        actual_home = sum(item.home_won for item in picks) / len(picks)
        by_league[league] = {
            "n": len(picks),
            "pickHomePct": round(picked_home * 100, 1),
            "actualHomeWinPct": round(actual_home * 100, 1),
            "biasPct": round((picked_home - actual_home) * 100, 1),
        }
    return by_league
