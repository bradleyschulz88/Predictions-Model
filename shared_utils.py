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


def win_pct_from_record(record: str | None, default: float = 0.5) -> float:
    """Convert a record string to win percentage.

    Handles formats like:
    - "10-5" -> wins=10, losses=5 -> 10/15 = 0.667
    - "10-5-2" -> wins=10, losses=5, draws=2 -> (10 + 0.5*2)/17 = 0.618
    """
    parsed = parse_record(record)
    if not parsed:
        return default
    if len(parsed) == 3:
        wins, draws, losses = parsed
        total = wins + draws + losses
        return (wins + 0.5 * draws) / total if total else default
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