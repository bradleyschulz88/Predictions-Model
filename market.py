"""Odds arithmetic: prices in, fair probabilities and expected value out.

Two things live here that the model was missing.

**De-vigging.** A bookmaker's prices imply probabilities that sum to more than
100%; the excess is their margin. Stripping it back to 100% recovers what the
market actually thinks. The obvious way -- divide each by the total -- assumes
the margin is spread evenly across both sides, and it is not: books load more
of it onto longshots. The power method solves for the exponent that makes the
prices sum to one instead, which corrects that skew.

**Expected value.** Percentage-point edge is not profit. Three points of edge
on a -300 favourite and three points on a +250 underdog are worth very
different amounts, because the payout differs. EV converts an edge and a price
into the only number that answers "is this bet worth making".
"""

from __future__ import annotations

from typing import Any, Sequence

# Below this the market is treated as unusable rather than extrapolated.
MIN_PRICE_PROB = 1e-4


def american_to_decimal(odds: float) -> float:
    """-155 -> 1.645 (stake 1 returns 1.645 including stake)."""
    value = float(odds)
    if value < 0:
        return 1.0 + 100.0 / abs(value)
    return 1.0 + value / 100.0


def american_to_implied(odds: float) -> float:
    """Raw implied probability, still carrying the bookmaker's margin."""
    value = float(odds)
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    return 100.0 / (value + 100.0)


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


# --------------------------------------------------------------------------
# De-vigging
# --------------------------------------------------------------------------


def devig_proportional(raw: Sequence[float]) -> list[float]:
    """Divide out the overround evenly. Simple, and biased toward longshots."""
    total = sum(raw)
    if total <= 0:
        return list(raw)
    return [value / total for value in raw]


def devig_power(raw: Sequence[float], *, tolerance: float = 1e-9) -> list[float]:
    """Solve for k where sum(p_i ** k) == 1, then return those probabilities.

    Books do not spread their margin evenly -- more of it sits on the longshot,
    which is the favourite-longshot bias. Raising every price to a common power
    shrinks the long tail harder than the short one, which is the right shape of
    correction. Falls back to proportional if the solve misbehaves.
    """
    prices = [max(MIN_PRICE_PROB, min(1.0 - MIN_PRICE_PROB, value)) for value in raw]
    if len(prices) < 2 or sum(prices) <= 1.0:
        return devig_proportional(prices)

    low, high = 0.5, 3.0
    for _ in range(80):
        mid = (low + high) / 2.0
        total = sum(price ** mid for price in prices)
        if abs(total - 1.0) < tolerance:
            break
        # Larger k shrinks every term, so the sum falls as k rises.
        if total > 1.0:
            low = mid
        else:
            high = mid

    solved = [price ** mid for price in prices]
    total = sum(solved)
    if total <= 0 or not all(0.0 < value < 1.0 for value in solved):
        return devig_proportional(prices)
    return [value / total for value in solved]


def devig(raw: Sequence[float], *, method: str = "power") -> list[float]:
    return devig_power(raw) if method == "power" else devig_proportional(raw)


# --------------------------------------------------------------------------
# Expected value
# --------------------------------------------------------------------------


def expected_value(probability: float, american_odds: float) -> float:
    """Profit per 1 unit staked, as a fraction. +0.05 means +5% per unit.

    This is the number that decides whether a bet is worth making. An edge in
    percentage points cannot, because it ignores what the bet pays.
    """
    decimal_odds = american_to_decimal(american_odds)
    win_profit = decimal_odds - 1.0
    return probability * win_profit - (1.0 - probability)


def kelly_fraction(probability: float, american_odds: float) -> float:
    """Kelly stake as a fraction of bankroll. Zero when there is no edge.

    Full Kelly is famously aggressive and assumes the probability is exactly
    right, which it never is; callers should scale this down.
    """
    win_profit = american_to_decimal(american_odds) - 1.0
    if win_profit <= 0:
        return 0.0
    fraction = (probability * win_profit - (1.0 - probability)) / win_profit
    return max(0.0, fraction)


def break_even_probability(american_odds: float) -> float:
    """The win rate a price needs just to break even."""
    return 1.0 / american_to_decimal(american_odds)


def assess_price(
    probability: float | None,
    american_odds: float | None,
    *,
    kelly_multiplier: float = 0.25,
    kelly_probability: float | None = None,
) -> dict[str, Any] | None:
    """Everything worth knowing about backing one side at one price.

    ``kelly_multiplier`` defaults to quarter Kelly: the model's probabilities
    are estimates with real error bars, and full Kelly on an overestimated edge
    is how bankrolls die.

    ``kelly_probability`` sizes the stake only -- it never touches the
    displayed confidence, edge or EV, which stay anchored to what the model
    actually said. Pass it when there is a graded calibration band for this
    exact confidence level: the measured win rate at that level is a better
    estimate of the true probability than the model's own number, which is
    exactly the case where stake size should move even though the headline
    number does not. Omit it (the default) to size off the model's stated
    probability, unchanged.
    """
    if probability is None or american_odds is None:
        return None
    try:
        odds = float(american_odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None

    value = expected_value(probability, odds)
    break_even = break_even_probability(odds)
    kelly_prob = probability if kelly_probability is None else kelly_probability
    full_kelly = kelly_fraction(kelly_prob, odds)

    return {
        "odds": int(odds),
        "decimalOdds": round(american_to_decimal(odds), 3),
        "modelPct": round(probability * 100, 1),
        "breakEvenPct": round(break_even * 100, 1),
        # Percentage points of probability above what the price requires.
        "edgePct": round((probability - break_even) * 100, 1),
        "evPct": round(value * 100, 2),
        "kellyPct": round(full_kelly * kelly_multiplier * 100, 2),
        # What the stake was actually sized against -- equal to modelPct
        # unless a calibration band overrode it.
        "kellyProbabilityPct": round(kelly_prob * 100, 1),
        "positive": value > 0,
    }
