"""Shared utility functions to avoid circular imports."""

from __future__ import annotations

import re


def parse_record(summary: str | None) -> tuple[int, ...] | None:
    """Parse a record string like '10-5' or '10-5-2' into (wins, losses, draws)."""
    if not summary:
        return None
    parts = re.split(r"[-\s]+", summary.strip())
    try:
        nums = tuple(int(p) for p in parts if p.isdigit())
        if len(nums) >= 2:
            return nums[:3]
    except ValueError:
        pass
    return None


# Three-part records are ambiguous across sports, and the middle number is the
# one that changes meaning: ESPN reports soccer as W-D-L (the standard league
# table order) but NFL/AFL as W-L-T. Only the position of the half-credit column
# differs -- the total is the same either way -- but reading "10-5-2" as W-D-L
# gives .735 and as W-L-T gives .647, so the caller has to say which it is.
WDL_LEAGUES = {"epl"}


def win_pct_from_record(
    record: str | None,
    default: float = 0.5,
    *,
    league: str | None = None,
) -> float:
    """Convert a record string to win percentage, counting draws as half a win.

    Handles formats like:
    - "10-5" -> 10 wins, 5 losses -> 10/15 = 0.667
    - "10-5-2" as W-D-L (soccer) -> (10 + 0.5*5) / 17 = 0.735
    - "10-5-2" as W-L-T (NFL/AFL) -> (10 + 0.5*2) / 17 = 0.647

    ``league`` selects the convention; it defaults to W-D-L because that is the
    format ESPN returns for the soccer competitions where three-part records
    actually occur in volume.
    """
    parsed = parse_record(record)
    if not parsed:
        return default
    if len(parsed) == 3:
        first, middle, last = parsed
        total = first + middle + last
        if not total:
            return default
        drawn = middle if (league is None or league in WDL_LEAGUES) else last
        return (first + 0.5 * drawn) / total
    wins, losses = parsed[:2]
    total = wins + losses
    return wins / total if total else default


def format_record(wins: int, losses: int, draws: int = 0) -> str:
    """Format wins, losses, draws into a record string."""
    if draws:
        return f"{wins}-{losses}-{draws}"
    return f"{wins}-{losses}"


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value between low and high."""
    return max(low, min(high, value))


def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero."""
    return num / denom if denom != 0 else default


def format_win_pct(record: str | None) -> str:
    """Format a record string as a win percentage string (e.g., '66.7%')."""
    pct = win_pct_from_record(record)
    return f"{pct * 100:.1f}%"