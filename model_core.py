"""Probability resolution for scheduled games.

Separates *what the probability is* from *how it is explained*. mlb_predictions
still builds the reasoning and factor text; this module decides the numbers.

Two paths:

1. A fitted logistic model (model_fit.py), trained on graded outcomes. Its
   output is already a maximum-likelihood estimate of the home win probability,
   so it needs no further shrinking.
2. A heuristic logit fallback, used before enough games have been graded to fit
   anything. This is the old hand-tuned path and it *is* overconfident, so the
   legacy bucket calibration still applies to it.

Which path ran is recorded on the result, because a probability that came from a
fallback should not be presented with the same authority as a fitted one.
"""

from __future__ import annotations

from typing import Any

from calibration_params import apply_platt, load_platt_params, platt_params_for
from data_providers.league_metrics import soccer_draw_probability
from model_fit import MAX_PROB, MIN_PROB, LogisticModel, load_model

_FITTED_MODEL: LogisticModel | None = None
_FITTED_MODEL_LOADED = False
_PLATT_PARAMS: dict[str, Any] | None = None


def get_platt_params() -> dict[str, Any]:
    """Load calibration curves once per process."""
    global _PLATT_PARAMS
    if _PLATT_PARAMS is None:
        _PLATT_PARAMS = {"plattParams": load_platt_params()}
    return _PLATT_PARAMS


def get_fitted_model() -> LogisticModel | None:
    """Load the fitted weights once per process; None when unavailable."""
    global _FITTED_MODEL, _FITTED_MODEL_LOADED
    if not _FITTED_MODEL_LOADED:
        _FITTED_MODEL = load_model()
        _FITTED_MODEL_LOADED = True
    return _FITTED_MODEL


def reset_fitted_model_cache() -> None:
    """Test hook: force the next call to re-read the weights and curves."""
    global _FITTED_MODEL, _FITTED_MODEL_LOADED, _PLATT_PARAMS
    _FITTED_MODEL, _FITTED_MODEL_LOADED = None, False
    _PLATT_PARAMS = None


def clamp(value: float, low: float = MIN_PROB, high: float = MAX_PROB) -> float:
    return max(low, min(high, value))


def resolve_probabilities(
    *,
    game: dict[str, Any],
    model_inputs: dict[str, Any],
    heuristic_home: float,
    enrichment: dict[str, Any],
    league_config: Any,
    league: str,
    legacy_calibrate: Any,
) -> dict[str, Any]:
    """Return the published outcome probabilities and how they were derived.

    ``heuristic_home`` is the old hand-tuned sigmoid, used only as a fallback.
    ``legacy_calibrate`` is the bucket-shrinkage callable, applied only on the
    fallback path -- shrinking a fitted probability would decalibrate it.
    """
    model = get_fitted_model()
    fitted_home = model.predict_proba(model_inputs, league) if model else None

    if fitted_home is not None:
        raw_binary_home = clamp(fitted_home)
        # A Platt curve fitted against *this* method's own graded history, in
        # home-probability space so the mapping stays symmetric. It is the
        # identity until enough fitted-path picks have been graded, so a curve
        # learned from the heuristic's mistakes is never applied here.
        curve = platt_params_for(get_platt_params(), method="fitted", league=league)
        binary_home = clamp(apply_platt(raw_binary_home, curve))
        method = "fitted"
        detail = "Logistic model fitted on graded outcomes"
    else:
        raw_binary_home = clamp(heuristic_home)
        # The heuristic stacks correlated features, so it needs the shrinkage.
        binary_home = clamp(
            legacy_calibrate(
                raw_binary_home,
                league=league,
                confidence_pct=max(raw_binary_home, 1.0 - raw_binary_home) * 100.0,
            )
        )
        method = "heuristic"
        detail = "Hand-tuned fallback (no fitted weights available)"

    binary_away = 1.0 - binary_home

    draw_prob = 0.0
    if league_config.supports_draw:
        # Split the binary home/away estimate across three outcomes. The draw is
        # *not* run through two-class shrinkage: pulling a 26% draw toward 50%
        # inflated it every time the old path renormalised.
        draw_prob = min(
            0.32,
            max(
                0.08,
                soccer_draw_probability(
                    league=league,
                    home_true=binary_home,
                    away_true=binary_away,
                    enrichment=enrichment,
                ),
            ),
        )
        scale = 1.0 - draw_prob
        home_prob = binary_home * scale
        away_prob = binary_away * scale
    else:
        home_prob, away_prob = binary_home, binary_away

    return {
        "home": home_prob,
        "away": away_prob,
        "draw": draw_prob if league_config.supports_draw else None,
        "binaryHome": binary_home,
        # Pre-calibration home probability. Logged so the next calibration fit
        # reads the model's raw output rather than its own previous corrections.
        "rawBinaryHome": raw_binary_home,
        "method": method,
        "detail": detail,
        "fittedAvailable": fitted_home is not None,
    }


def model_metadata() -> dict[str, Any]:
    """Describe the active model for display and debugging."""
    model = get_fitted_model()
    if model is None:
        return {"method": "heuristic", "fitted": False}
    metadata = dict(model.metadata)
    metadata.update({"method": "fitted", "fitted": True})
    walk_forward = model.to_dict().get("walkForward") or {}
    if walk_forward.get("logLoss") is not None:
        metadata["walkForward"] = walk_forward
    return metadata
