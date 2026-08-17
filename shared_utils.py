"""Shared utility functions to avoid circular imports."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


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


def games_played(record: str | None) -> int:
    """How many games are behind a record string, or 0 when there are none.

    The sample size a win percentage rests on, which the model had no way to
    ask for. A club at 1-0 and a club at 12-4 both produced a win percentage
    and nothing distinguished them, so one preseason result was read with the
    same authority as a third of a season.
    """
    parsed = parse_record(record)
    if not parsed:
        return 0
    return sum(parsed[:3]) if len(parsed) == 3 else sum(parsed[:2])


def win_pct_or_none(record: str | None, *, league: str | None = None) -> float | None:
    """Win percentage, or None when the club has not played yet.

    ``win_pct_from_record`` folds "0-0" and "no record at all" into its 0.5
    default, which is right for display -- a blank cell needs *some* number --
    and wrong for a model input. A team that is 0-0 is not a .500 team; it is
    a team with no evidence either way, and the difference matters most in the
    first weeks of a season when every club is 0-0 at once.

    Feeding the 0.5 default into a difference produces exactly 0.0, which then
    passes through splitDiff's home-field centring and comes out as a small
    negative -- a phantom edge against the home side on every opening-week
    game. Returning None instead lets the fit standardise the feature to its
    training mean, which is what "absent" is supposed to mean everywhere else
    in this model.
    """
    parsed = parse_record(record)
    if not parsed:
        return None
    total = sum(parsed[:3]) if len(parsed) == 3 else sum(parsed[:2])
    if not total:
        return None
    return win_pct_from_record(record, league=league)


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


# --------------------------------------------------------------------------
# Writing JSON the browser can actually read
# --------------------------------------------------------------------------
#
# `json.dumps` emits the bare literals `NaN`, `Infinity` and `-Infinity` by
# default. None of the three is JSON: `JSON.parse` throws on the first one it
# meets. So a single non-finite float anywhere in a payload does not render a
# dash where a number should be -- it takes down the whole page that reads the
# file, because nothing after the throw runs.
#
# The build had twelve separate `json.dumps` calls and no shared writer, so
# that failure was one arithmetic accident away in twelve places. It is
# reachable: `_accumulate_summary` sums `float(item.get("units") or 0.0)` over
# stored rows, and a NaN in any one of them poisons the bucket total, the ROI
# derived from it, and then `accuracy.json`. NaN also round-trips -- Python's
# loader accepts the literal it wrote -- so a bad value persists across builds
# rather than clearing on the next one.
#
# Replacing the value with null loses one figure and the page shows a dash.
# That is the right trade against losing the page.


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None. Everything else passes."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dumps_json(payload: Any, *, indent: int | None = 2, default: Any = str) -> str:
    """`json.dumps` that cannot emit a document the browser refuses to parse.

    `allow_nan=False` is kept on deliberately after the sanitising pass. It
    should now be unreachable, and if it ever raises, the sanitiser missed a
    path -- which is worth a loud failure in a build rather than a quiet blank
    page for a reader.
    """
    return json.dumps(json_safe(payload), indent=indent, default=default, allow_nan=False)


def write_json(path: Path | str, payload: Any, *, indent: int | None = 2, default: Any = str) -> None:
    """Write a payload as JSON that `JSON.parse` will accept."""
    Path(path).write_text(dumps_json(payload, indent=indent, default=default), encoding="utf-8")