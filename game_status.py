"""Normalize ESPN scoreboard status into dashboard-friendly flags."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

STATUS_IN_PROGRESS = "STATUS_IN_PROGRESS"
STATUS_FINAL = "STATUS_FINAL"
STATUS_SCHEDULED = "STATUS_SCHEDULED"
STATUS_POSTPONED = "STATUS_POSTPONED"

NON_PLAYABLE_NAMES = frozenset(
    {
        STATUS_POSTPONED,
        "STATUS_CANCELED",
        "STATUS_CANCELLED",
        "STATUS_SUSPENDED",
        "STATUS_DELAYED",
        "STATUS_FORFEIT",
    }
)

# If ESPN still says in-progress but nobody is in the park after this many minutes, treat as washed out.
STALE_LIVE_MINUTES = 20
# Faster cutoff when the scoreboard is still 0-0 with no attendance (typical pre-game rainout).
STALE_SCORELESS_MINUTES = 10


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _note_text(notes: list[Any] | None) -> str:
    parts: list[str] = []
    for note in notes or []:
        if not isinstance(note, dict):
            continue
        for key in ("headline", "text", "description"):
            value = note.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts)


def _total_runs(home_score: Any, away_score: Any) -> int | None:
    try:
        if home_score is None or away_score is None:
            return None
        return int(home_score) + int(away_score)
    except (TypeError, ValueError):
        return None


def _is_washed_out_in_progress(
    *,
    name: str,
    state: str,
    attendance: int | None,
    total_runs: int | None,
    start_date: str | None,
    now: datetime | None = None,
) -> bool:
    """ESPN rainouts often stay STATUS_IN_PROGRESS with 0-0 and attendance 0."""
    if name != STATUS_IN_PROGRESS or state != "in":
        return False
    if attendance not in (0, None):
        return False
    if total_runs not in (None, 0):
        return False
    started = parse_iso_datetime(start_date)
    if not started:
        return False
    reference = now or datetime.now(timezone.utc)
    elapsed_minutes = (reference - started).total_seconds() / 60
    return elapsed_minutes >= STALE_SCORELESS_MINUTES


def normalize_espn_status(
    status_type: dict[str, Any],
    *,
    start_date: str | None = None,
    attendance: int | None = None,
    notes: list[Any] | None = None,
    home_score: Any = None,
    away_score: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Map ESPN status.type (+ context) to isLive / isFinal / void flags."""
    name = str(status_type.get("name") or "")
    state = str(status_type.get("state") or "")
    completed = bool(status_type.get("completed"))
    description = status_type.get("description") or status_type.get("shortDetail") or "Scheduled"
    detail = status_type.get("detail") or status_type.get("shortDetail") or description

    blob = " ".join([name, description, detail, _note_text(notes)]).lower()

    is_postponed = name == STATUS_POSTPONED or "postpon" in blob
    is_canceled = name in ("STATUS_CANCELED", "STATUS_CANCELLED") or "canceled" in blob or "cancelled" in blob
    is_suspended = name == "STATUS_SUSPENDED" or "suspend" in blob
    is_delayed = name == "STATUS_DELAYED" or "delay" in blob
    is_voided = is_postponed or is_canceled

    is_final = completed or name == STATUS_FINAL
    is_scheduled = name == STATUS_SCHEDULED or state == "pre"

    is_live = name == STATUS_IN_PROGRESS and state == "in" and not is_voided and not is_suspended
    is_washed_out = False
    total_runs = _total_runs(home_score, away_score)
    started = parse_iso_datetime(start_date)
    reference = now or datetime.now(timezone.utc)
    elapsed_minutes = (reference - started).total_seconds() / 60 if started else None

    if is_live and _is_washed_out_in_progress(
        name=name,
        state=state,
        attendance=attendance,
        total_runs=total_runs,
        start_date=start_date,
        now=reference,
    ):
        is_live = False
        is_washed_out = True
        is_delayed = True
        is_voided = True
    elif (
        is_live
        and attendance == 0
        and elapsed_minutes is not None
        and elapsed_minutes >= STALE_LIVE_MINUTES
        and total_runs is not None
        and total_runs <= 1
    ):
        # Rainouts often linger as IN_PROGRESS with 0-1 total runs and attendance still 0.
        is_live = False
        is_washed_out = True
        is_delayed = True
        is_voided = True
    elif is_live and attendance == 0 and elapsed_minutes is not None:
        if elapsed_minutes >= STALE_LIVE_MINUTES * 2 and total_runs not in (None, 0):
            is_live = False
            is_delayed = True

    if not is_live and name == STATUS_IN_PROGRESS and state == "in":
        is_washed_out = is_washed_out or _is_washed_out_in_progress(
            name=name,
            state=state,
            attendance=attendance,
            total_runs=total_runs,
            start_date=start_date,
            now=reference,
        ) or (
            attendance == 0
            and elapsed_minutes is not None
            and elapsed_minutes >= STALE_LIVE_MINUTES
            and total_runs is not None
            and total_runs <= 1
        )
        if is_washed_out:
            is_delayed = True
            is_voided = True

    game_status_text = str(description)
    if is_postponed:
        game_status_text = "Postponed"
    elif is_canceled:
        game_status_text = "Canceled"
    elif is_suspended:
        game_status_text = "Suspended"
    elif is_washed_out:
        game_status_text = "Washed out"
    elif is_delayed and not is_live:
        game_status_text = "Delayed"

    return {
        "statusType": name,
        "gameStatusText": game_status_text,
        "gameStatusDetail": str(detail),
        "isLive": is_live,
        "isFinal": is_final,
        "isScheduled": is_scheduled,
        "isPostponed": is_postponed,
        "isCanceled": is_canceled,
        "isSuspended": is_suspended,
        "isDelayed": is_delayed,
        "isVoided": is_voided or is_washed_out,
        "isWashedOut": is_washed_out,
        "attendance": attendance,
    }
