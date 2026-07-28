"""Fit win-probability coefficients from graded outcomes (stdlib only).

The scoring model previously used hand-picked multipliers (record x3.2, splits
x2.4, power x2.2, form x1.8, ...) applied to features that measure largely the
same thing -- recordDiff and powerDiff correlate at 0.90 -- so one signal was
counted three or four times and the resulting probabilities were far too
confident. This module replaces the guesswork: it collapses the collinear
inputs into a single strength score and fits every coefficient by penalised
maximum likelihood against games that have actually been played.

Deliberately small: with a few hundred graded games per league, a handful of
regularised features is the most a fit can support without memorising noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

WEIGHTS_FILE = "model_weights.json"

# Feature order is part of the serialised model; append, never reorder.
#
# Chosen by walk-forward ablation (`python model_fit.py --ablate`), not by taste.
# Every additional feature measured worse out of sample on the 675 graded games
# available:
#
#   strengthDiff                    logloss 0.6443
#   + marketLogit                   logloss 0.6416   <- selected
#   + pitchingDiff                  logloss 0.6447
#   + restDiff                      logloss 0.6522
#   + injuryDiff                    logloss 0.6557
#   + b2bDiff                       logloss 0.6552
#
# That is a statement about these encodings at this sample size, not about the
# real world. Starting pitching genuinely matters in baseball -- but the market
# has already priced it, so once marketLogit is in the model, adding starter and
# bullpen ERA on top is redundant rather than additive.
#
# restDiff and b2bDiff carry an ASTERISK. Between 2026-07-23 and 2026-07-28,
# apply_predictions re-ran enrichment per game without a schedule context,
# which overwrote every rest day and back-to-back flag with None/False. Roughly
# the last 200 graded games therefore have no rest data at all, so the ablation
# above scored those features on partly destroyed input and cannot be treated
# as a fair verdict on them. Re-run it once a couple of weeks of post-fix games
# have graded, and promote them here if the answer changes.
#
# The enrichment pipeline still supplies all of them to the reasoning panel;
# they are simply not allowed to move the probability until they can earn it.
ANCHORED_FEATURES = ("strengthDiff", "marketLogit")
STANDALONE_FEATURES = ("strengthDiff",)

# Every feature the collapser knows how to build, for ablation runs.
CANDIDATE_FEATURES = ("strengthDiff", "marketLogit", "pitchingDiff", "restDiff", "injuryDiff", "b2bDiff")

# Shrinkage constant for per-league intercepts: a league needs ~K graded games
# before its own home-field estimate outweighs the pooled one.
LEAGUE_INTERCEPT_PRIOR = 50.0

# Probabilities are never published outside this band. Baseball in particular
# has no 95% games; the old 0.05/0.95 clamp let stacked features run to the rail.
MIN_PROB = 0.05
MAX_PROB = 0.95

# Fallback used only before a fit has measured the real value from the log.
DEFAULT_SPLIT_DIFF_CENTRE = 0.0457


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def logit(prob: float) -> float:
    prob = max(1e-6, min(1.0 - 1e-6, prob))
    return math.log(prob / (1.0 - prob))


# --------------------------------------------------------------------------
# Linear algebra (small dense systems only)
# --------------------------------------------------------------------------


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot
            if factor:
                for col in range(column, size + 1):
                    augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, size)
        )
        solution[row] = total / augmented[row][row]
    return solution


# --------------------------------------------------------------------------
# Penalised logistic regression
# --------------------------------------------------------------------------


def fit_logistic(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> list[float]:
    """IRLS/Newton fit. Row vectors must already include a leading 1.0 intercept.

    The intercept is left unpenalised so regularisation shrinks effects toward
    zero without dragging the base rate away from the data.
    """
    if not rows:
        return []
    width = len(rows[0])
    weights = [0.0] * width

    for _ in range(max_iterations):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]

        for row, label in zip(rows, labels):
            prediction = sigmoid(sum(w * x for w, x in zip(weights, row)))
            residual = label - prediction
            # Floor the weight so a saturated fit cannot produce a singular Hessian.
            variance = max(prediction * (1.0 - prediction), 1e-8)
            for i in range(width):
                gradient[i] += residual * row[i]
                row_i_var = row[i] * variance
                for j in range(width):
                    hessian[i][j] += row_i_var * row[j]

        for i in range(width):
            penalty = 0.0 if i == 0 else l2
            gradient[i] -= penalty * weights[i]
            hessian[i][i] += penalty

        step = solve_linear(hessian, gradient)
        if step is None:
            break
        weights = [w + s for w, s in zip(weights, step)]
        if max(abs(s) for s in step) < tolerance:
            break

    return weights


def predict_row(weights: Sequence[float], row: Sequence[float]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, row)))


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def build_feature_dict(
    features: dict[str, Any] | None,
    *,
    split_diff_centre: float | None = None,
) -> dict[str, float | None]:
    """Collapse the logged feature blob into the small set the model fits on.

    recordDiff, splitDiff and powerDiff correlate at 0.82-0.90 -- they are three
    measurements of team quality, not three signals. Averaging whichever are
    present yields one input, so the fit assigns team strength a single
    coefficient instead of stacking three.
    """
    features = features or {}
    centre = DEFAULT_SPLIT_DIFF_CENTRE if split_diff_centre is None else split_diff_centre

    strength_parts: list[float] = []
    record_diff = _first_number(features.get("recordDiff"))
    if record_diff is not None:
        strength_parts.append(record_diff)
    split_diff = _first_number(features.get("splitDiff"))
    if split_diff is not None:
        # Home/road splits embed home-field advantage. Centring on the observed
        # league-average gap leaves only team-specific home strength, so the
        # fitted intercept carries home field exactly once.
        strength_parts.append(split_diff - centre)
    home_power = _first_number(features.get("homePower"))
    away_power = _first_number(features.get("awayPower"))
    if home_power is not None and away_power is not None:
        strength_parts.append(home_power - away_power)

    strength = sum(strength_parts) / len(strength_parts) if strength_parts else None

    implied_home = _first_number(features.get("impliedHome"))
    market_logit = logit(implied_home / 100.0) if implied_home is not None else None

    home_rest = _first_number(features.get("homeRest"))
    away_rest = _first_number(features.get("awayRest"))
    rest_diff = (home_rest - away_rest) if home_rest is not None and away_rest is not None else None

    home_injury = _first_number(features.get("homeInjuryLoad")) or 0.0
    away_injury = _first_number(features.get("awayInjuryLoad")) or 0.0

    home_b2b = 1.0 if features.get("homeBackToBack") else 0.0
    away_b2b = 1.0 if features.get("awayBackToBack") else 0.0

    return {
        "strengthDiff": strength,
        "marketLogit": market_logit,
        "pitchingDiff": _pitching_diff(features),
        "restDiff": rest_diff,
        "injuryDiff": away_injury - home_injury,
        "b2bDiff": away_b2b - home_b2b,
    }


# The starter throws roughly two thirds of a game and the bullpen the rest, so
# the run-prevention edge is weighted accordingly. Sign convention matches the
# rest: positive favours the home side.
STARTER_WEIGHT = 0.65
BULLPEN_WEIGHT = 0.35


def _pitching_diff(features: dict[str, Any]) -> float | None:
    """Run-prevention edge in ERA units, home minus away.

    MLB-only, and None elsewhere so it contributes nothing to other leagues.
    Uses ERA rather than FIP because FIP is absent from the logged history while
    starter and bullpen ERA are present on 97-100% of games.
    """
    pitching = features.get("mlbPitching")
    if not isinstance(pitching, dict):
        return None

    parts: list[tuple[float, float]] = []

    home_starter = _first_number(
        pitching.get("homePitcherRecentEra"), pitching.get("homePitcherApiEra")
    )
    away_starter = _first_number(
        pitching.get("awayPitcherRecentEra"), pitching.get("awayPitcherApiEra")
    )
    if home_starter is not None and away_starter is not None:
        parts.append((away_starter - home_starter, STARTER_WEIGHT))

    home_bullpen = _first_number(pitching.get("homeBullpenEra"))
    away_bullpen = _first_number(pitching.get("awayBullpenEra"))
    if home_bullpen is not None and away_bullpen is not None:
        parts.append((away_bullpen - home_bullpen, BULLPEN_WEIGHT))

    if not parts:
        return None
    weight_total = sum(weight for _, weight in parts)
    return sum(value * weight for value, weight in parts) / weight_total


def measure_split_diff_centre(features_list: Sequence[dict[str, Any]]) -> float:
    """Average home-record minus away-record gap across the training set.

    This is the portion of splitDiff that is simply home-field advantage rather
    than team quality, and it is what gets subtracted so the fitted intercept
    carries home field exactly once instead of twice.
    """
    values = [
        float(features["splitDiff"])
        for features in features_list
        if (features or {}).get("splitDiff") is not None
    ]
    return round(sum(values) / len(values), 4) if values else DEFAULT_SPLIT_DIFF_CENTRE


def to_row(
    values: dict[str, float | None],
    feature_names: Sequence[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[float]:
    """Standardised design row with a leading intercept.

    Missing values fall back to the training mean, which after centring is zero
    -- an absent feature therefore contributes nothing rather than silently
    reading as an extreme value.
    """
    row = [1.0]
    for name in feature_names:
        value = values.get(name)
        if value is None:
            row.append(0.0)
            continue
        scale = scales.get(name) or 1.0
        row.append((value - means.get(name, 0.0)) / scale)
    return row


def standardisation(
    samples: Sequence[dict[str, float | None]], feature_names: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        present = [row[name] for row in samples if row.get(name) is not None]
        if not present:
            means[name], scales[name] = 0.0, 1.0
            continue
        mean = sum(present) / len(present)
        variance = sum((value - mean) ** 2 for value in present) / len(present)
        means[name] = mean
        scales[name] = math.sqrt(variance) or 1.0
    return means, scales


# --------------------------------------------------------------------------
# Model object
# --------------------------------------------------------------------------


class ProbabilityModel(Protocol):
    """Seam for swapping in a stronger learner without touching the caller.

    ml_model/ holds a trained XGBoost + isotonic calibrator that is not on the
    prediction path. Once there is enough graded data to justify it, it can be
    wrapped to satisfy this protocol and dropped in behind the same interface.
    """

    def predict_proba(self, features: dict[str, Any], league: str) -> float | None:
        ...


class LogisticModel:
    """Fitted logistic model with per-league intercepts."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self._anchored = payload.get("anchored") or {}
        self._standalone = payload.get("standalone") or {}
        self._league_intercepts = payload.get("leagueIntercepts") or {}
        self.split_diff_centre = float(
            payload.get("splitDiffCentre", DEFAULT_SPLIT_DIFF_CENTRE)
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._payload.get("metadata") or {}

    def predict_proba(self, features: dict[str, Any], league: str) -> float | None:
        """Home win probability from a raw logged/live feature blob."""
        values = build_feature_dict(features, split_diff_centre=self.split_diff_centre)
        return self.predict_from_values(values, league)

    def predict_from_values(
        self, values: dict[str, float | None], league: str
    ) -> float | None:
        """Home win probability from an already-collapsed feature dict."""
        use_anchored = values.get("marketLogit") is not None and self._anchored.get("weights")
        block = self._anchored if use_anchored else self._standalone
        weights = block.get("weights")
        if not weights:
            return None

        names = block.get("features") or []
        row = to_row(values, names, block.get("means") or {}, block.get("scales") or {})
        score = sum(w * x for w, x in zip(weights, row))
        score += float(self._league_intercepts.get(league, 0.0))
        return max(MIN_PROB, min(MAX_PROB, sigmoid(score)))

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def load_model(data_dir: Path | None = None) -> LogisticModel | None:
    """Load fitted weights, returning None so callers can fall back cleanly."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    path = data_dir / WEIGHTS_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not payload.get("anchored") and not payload.get("standalone"):
        return None
    return LogisticModel(payload)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


class Sample:
    __slots__ = ("values", "label", "league", "date", "has_market")

    def __init__(self, values: dict[str, float | None], label: int, league: str, date: str) -> None:
        self.values = values
        self.label = label
        self.league = league
        self.date = date
        self.has_market = values.get("marketLogit") is not None


def samples_from_log(data_dir: Path) -> tuple[list[Sample], float]:
    """Join logged features to graded outcomes.

    Returns the samples plus the split-diff centre measured from this data, so
    the same value is used at fit time and at prediction time.
    """
    log = _load_json(data_dir / "predictions_log.json", {"predictions": {}})
    accuracy = _load_json(data_dir / "accuracy.json", {"picksByEventId": {}})

    raw: list[tuple[dict[str, Any], int, str, str]] = []
    for event_id, graded in (accuracy.get("picksByEventId") or {}).items():
        if graded.get("status") != "graded":
            continue
        logged = (log.get("predictions") or {}).get(event_id) or {}
        merged = {**logged, **graded}
        label = _home_label(merged)
        if label is None:
            continue
        raw.append(
            (
                merged.get("features") or {},
                label,
                merged.get("league") or "unknown",
                merged.get("date") or merged.get("scheduleDate") or "",
            )
        )

    centre = measure_split_diff_centre([features for features, _, _, _ in raw])
    samples = [
        Sample(
            values=build_feature_dict(features, split_diff_centre=centre),
            label=label,
            league=league,
            date=date,
        )
        for features, label, league, date in raw
    ]
    return samples, centre


def _home_label(record: dict[str, Any]) -> int | None:
    home_score, away_score = record.get("homeScore"), record.get("awayScore")
    if home_score is None or away_score is None:
        return None
    try:
        home, away = int(home_score), int(away_score)
    except (TypeError, ValueError):
        return None
    if home == away:
        return None  # Draws are handled separately, not as a home/away label.
    return 1 if home > away else 0


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _fit_block(
    samples: Sequence[Sample], feature_names: Sequence[str], *, l2: float
) -> dict[str, Any] | None:
    if len(samples) < len(feature_names) * 5:
        return None
    values = [sample.values for sample in samples]
    means, scales = standardisation(values, feature_names)
    rows = [to_row(sample.values, feature_names, means, scales) for sample in samples]
    labels = [sample.label for sample in samples]
    weights = fit_logistic(rows, labels, l2=l2)
    if not weights:
        return None
    return {
        "features": list(feature_names),
        "weights": weights,
        "means": means,
        "scales": scales,
        "n": len(samples),
        "l2": l2,
    }


def _league_intercepts(
    samples: Sequence[Sample], anchored: dict[str, Any] | None, standalone: dict[str, Any] | None
) -> dict[str, float]:
    """Per-league home-field correction, shrunk toward the pooled fit.

    A league with 46 graded games should barely move its own intercept; one with
    478 should move it most of the way.
    """
    by_league: dict[str, list[Sample]] = {}
    for sample in samples:
        by_league.setdefault(sample.league, []).append(sample)

    intercepts: dict[str, float] = {}
    for league, league_samples in by_league.items():
        residual_sum = 0.0
        count = 0
        for sample in league_samples:
            block = anchored if (sample.has_market and anchored) else standalone
            if not block:
                continue
            row = to_row(sample.values, block["features"], block["means"], block["scales"])
            predicted = predict_row(block["weights"], row)
            residual_sum += sample.label - predicted
            count += 1
        if not count:
            continue
        # One Newton step on the intercept alone, then shrink toward zero.
        mean_residual = residual_sum / count
        raw = mean_residual / 0.25  # 0.25 is the max of p(1-p); a conservative step.
        shrink = count / (count + LEAGUE_INTERCEPT_PRIOR)
        intercepts[league] = round(raw * shrink, 4)
    return intercepts


def fit_from_observations(
    samples: Sequence[Sample],
    *,
    l2: float = 1.0,
    split_diff_centre: float = DEFAULT_SPLIT_DIFF_CENTRE,
    anchored_features: Sequence[str] = ANCHORED_FEATURES,
    standalone_features: Sequence[str] = STANDALONE_FEATURES,
) -> dict[str, Any]:
    """Fit anchored and standalone blocks plus per-league intercepts."""
    anchored_samples = [sample for sample in samples if sample.has_market]
    anchored = _fit_block(anchored_samples, anchored_features, l2=l2)
    standalone = _fit_block(samples, standalone_features, l2=l2)
    intercepts = _league_intercepts(samples, anchored, standalone)

    return {
        "anchored": anchored or {},
        "standalone": standalone or {},
        "leagueIntercepts": intercepts,
        "splitDiffCentre": split_diff_centre,
        "metadata": {
            "nTotal": len(samples),
            "nWithMarket": len(anchored_samples),
            "l2": l2,
            "leagues": sorted({sample.league for sample in samples}),
        },
    }


def walk_forward_scores(
    samples: Sequence[Sample],
    *,
    l2: float = 1.0,
    folds: int = 5,
    anchored_features: Sequence[str] = ANCHORED_FEATURES,
    standalone_features: Sequence[str] = STANDALONE_FEATURES,
) -> dict[str, Any]:
    """Expanding-window evaluation: never score a game the fit has seen.

    Sports data is a time series and the model is retrained on its own history,
    so a random split would leak future information and flatter the model.
    """
    ordered = sorted(samples, key=lambda sample: (sample.date, sample.league))
    if len(ordered) < folds * 20:
        return {"folds": 0, "note": "insufficient data for walk-forward"}

    start = len(ordered) // (folds + 1)
    predictions: list[tuple[float, int]] = []
    for fold in range(folds):
        split = start * (fold + 1)
        train, test = ordered[:split], ordered[split : split + start]
        if not test:
            continue
        payload = fit_from_observations(
            train,
            l2=l2,
            anchored_features=anchored_features,
            standalone_features=standalone_features,
        )
        model = LogisticModel(payload)
        for sample in test:
            # Values are already collapsed; re-deriving them would re-centre
            # against a different training set and leak information.
            prob = model.predict_from_values(sample.values, sample.league)
            if prob is not None:
                predictions.append((prob, sample.label))

    if not predictions:
        return {"folds": 0}

    log_loss = -sum(
        math.log(max(1e-9, prob if label else 1.0 - prob)) for prob, label in predictions
    ) / len(predictions)
    brier = sum((prob - label) ** 2 for prob, label in predictions) / len(predictions)
    hits = sum(1 for prob, label in predictions if (prob >= 0.5) == bool(label))

    return {
        "folds": folds,
        "n": len(predictions),
        "logLoss": round(log_loss, 4),
        "brier": round(brier, 4),
        "accuracy": round(hits / len(predictions), 4),
    }


def choose_l2(
    samples: Sequence[Sample],
    candidates: Iterable[float] = (0.3, 1.0, 3.0, 10.0, 30.0),
    **feature_kwargs: Any,
) -> float:
    """Pick the ridge strength with the best walk-forward log loss."""
    best_l2, best_loss = 1.0, float("inf")
    for candidate in candidates:
        scores = walk_forward_scores(samples, l2=candidate, **feature_kwargs)
        loss = scores.get("logLoss")
        if loss is not None and loss < best_loss:
            best_l2, best_loss = candidate, loss
    return best_l2


def ablate(samples: Sequence[Sample]) -> list[dict[str, Any]]:
    """Walk-forward score of each nested feature set, best ridge per set.

    Keeps feature selection honest and repeatable: as the graded log grows, a
    feature that cannot beat its own absence should not be in the model.
    """
    results: list[dict[str, Any]] = []
    for size in range(1, len(CANDIDATE_FEATURES) + 1):
        anchored = CANDIDATE_FEATURES[:size]
        standalone = tuple(name for name in anchored if name != "marketLogit")
        best: dict[str, Any] | None = None
        for l2 in (1.0, 3.0, 10.0, 30.0):
            scores = walk_forward_scores(
                samples,
                l2=l2,
                anchored_features=anchored,
                standalone_features=standalone,
            )
            if scores.get("logLoss") is None:
                continue
            if best is None or scores["logLoss"] < best["logLoss"]:
                best = {**scores, "l2": l2}
        if best:
            results.append({"features": list(anchored), **best})
    return results


def fit_and_write(data_dir: Path, *, l2: float | None = None) -> dict[str, Any]:
    """Fit from the graded log and persist weights next to the other data files."""
    samples, centre = samples_from_log(data_dir)
    if not samples:
        raise SystemExit("No graded samples found; cannot fit.")

    chosen_l2 = choose_l2(samples) if l2 is None else l2
    payload = fit_from_observations(samples, l2=chosen_l2, split_diff_centre=centre)
    payload["walkForward"] = walk_forward_scores(samples, l2=chosen_l2)
    (data_dir / WEIGHTS_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _describe(payload: dict[str, Any]) -> None:
    print(f"Fitted on {payload['metadata']['nTotal']} graded games"
          f" ({payload['metadata']['nWithMarket']} with odds), l2={payload['metadata']['l2']}")
    print(f"Split-diff centre (home-field baked into splits): {payload['splitDiffCentre']:+.4f}\n")

    for name in ("anchored", "standalone"):
        block = payload.get(name) or {}
        if not block.get("weights"):
            print(f"  {name}: not fitted (insufficient data)")
            continue
        print(f"  {name} (n={block['n']})")
        weights = block["weights"]
        print(f"    {'intercept':<14} {weights[0]:+.4f}   <- home-field advantage")
        for feature, weight in zip(block["features"], weights[1:]):
            print(f"    {feature:<14} {weight:+.4f}")
        print()

    print("  per-league intercept corrections (shrunk toward pooled)")
    for league, value in sorted((payload.get("leagueIntercepts") or {}).items()):
        print(f"    {league:<10} {value:+.4f}")

    walk = payload.get("walkForward") or {}
    if walk.get("n"):
        print(f"\n  walk-forward (n={walk['n']}, {walk['folds']} folds):"
              f" logloss {walk['logLoss']} · brier {walk['brier']} · acc {walk['accuracy']}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit win-probability coefficients from graded games.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "docs" / "data",
    )
    parser.add_argument("--l2", type=float, default=None, help="Ridge strength (default: choose by walk-forward)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and report without writing weights")
    parser.add_argument("--ablate", action="store_true", help="Walk-forward score each nested feature set")
    args = parser.parse_args()

    if args.ablate:
        samples, _ = samples_from_log(args.data_dir)
        print(f"Walk-forward ablation on {len(samples)} graded games")
        print(f"  {'features':<52} {'l2':>5} {'logloss':>8} {'brier':>8} {'acc':>7}")
        for row in ablate(samples):
            names = "+".join(row["features"])
            print(f"  {names:<52} {row['l2']:>5} {row['logLoss']:>8.4f}"
                  f" {row['brier']:>8.4f} {row['accuracy']:>7.4f}")
        return 0

    if args.dry_run:
        samples, centre = samples_from_log(args.data_dir)
        chosen = choose_l2(samples) if args.l2 is None else args.l2
        payload = fit_from_observations(samples, l2=chosen, split_diff_centre=centre)
        payload["walkForward"] = walk_forward_scores(samples, l2=chosen)
    else:
        payload = fit_and_write(args.data_dir, l2=args.l2)

    _describe(payload)
    if not args.dry_run:
        print(f"\nWrote {args.data_dir / WEIGHTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
