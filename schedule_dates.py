"""League-aware schedule dates (ESPN uses the league's local calendar day)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sports_config import get_league

LEAGUE_TIMEZONES: dict[str, str] = {
    "mlb": "America/New_York",
    "nfl": "America/New_York",
    "nba": "America/New_York",
    "worldcup": "America/New_York",
    "epl": "Europe/London",
    "afl": "Australia/Melbourne",
}

# US sports: before this hour (league local), yesterday's slate may still be live.
EARLY_SLATE_CUTOFF_HOUR = 10


def get_schedule_timezone(league: str) -> str:
    return LEAGUE_TIMEZONES.get(league, "UTC")


def league_now(league: str) -> datetime:
    return datetime.now(ZoneInfo(get_schedule_timezone(league)))


def league_schedule_date(league: str, days_offset: int = 0) -> str:
    return (league_now(league).date() + timedelta(days=days_offset)).isoformat()


def default_game_date(league: str = "mlb") -> str:
    """Default ESPN scoreboard date in the league's schedule timezone."""
    get_league(league)
    now = league_now(league)
    today = now.date()

    if league in {"mlb", "nfl", "nba"} and now.hour < EARLY_SLATE_CUTOFF_HOUR:
        return (today - timedelta(days=1)).isoformat()

    return today.isoformat()


def schedule_dates_for_league(league: str) -> list[str]:
    """Candidate dates to pre-build or show in the date picker."""
    default = default_game_date(league)
    candidates = [
        default,
        league_schedule_date(league, 0),
        league_schedule_date(league, 1),
        league_schedule_date(league, -1),
    ]

    if league in {"mlb", "nfl", "nba"}:
        candidates.append(league_schedule_date(league, -2))

    ordered: list[str] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return sorted(ordered)


def schedule_date_label(league: str, iso_date: str) -> str:
    tz = get_schedule_timezone(league)
    today = league_schedule_date(league, 0)
    tomorrow = league_schedule_date(league, 1)
    yesterday = league_schedule_date(league, -1)
    parsed = date.fromisoformat(iso_date)
    formatted = parsed.strftime("%a %b ") + str(parsed.day)

    if iso_date == today:
        prefix = "Schedule today"
    elif iso_date == tomorrow:
        prefix = "Schedule tomorrow"
    elif iso_date == yesterday:
        prefix = "Schedule yesterday"
    else:
        prefix = "Schedule"

    return f"{prefix} · {formatted} ({tz})"
