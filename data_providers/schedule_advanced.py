"""Schedule-derived flags: back-to-back, three-in-four, travel proxy."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from espn_client import fetch_scoreboard, parse_scoreboard

_ROLLING_CACHE: dict[tuple[str, str, int], list[dict[str, Any]]] = {}


def _parse_start(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _team_game_dates(games: list[dict[str, Any]], team_name: str | None) -> list[datetime]:
    if not team_name:
        return []
    dates: list[datetime] = []
    for game in games:
        if team_name not in {game.get("homeTeam"), game.get("awayTeam")}:
            continue
        start = _parse_start(game.get("startDate"))
        if start:
            dates.append(start)
    return sorted(dates)


def compute_schedule_flags(
    games: list[dict[str, Any]],
    team_name: str | None,
    current_start: str | None,
) -> dict[str, Any]:
    current = _parse_start(current_start)
    if not current:
        return {}

    prior = [dt for dt in _team_game_dates(games, team_name) if dt < current]
    if not prior:
        return {"restDays": None, "backToBack": False, "threeInFour": False}

    last = prior[-1]
    rest_days = max(0, (current.date() - last.date()).days)
    back_to_back = rest_days <= 1
    four_day_window = [dt for dt in prior if (current.date() - dt.date()).days <= 3]
    three_in_four = len(four_day_window) >= 2

    return {
        "restDays": rest_days,
        "backToBack": back_to_back,
        "threeInFour": three_in_four,
        "gamesLast4Days": len(four_day_window) + 1,
    }


def schedule_flags_logit_adjustment(enrichment: dict[str, Any]) -> float:
    home = enrichment.get("homeScheduleFlags") or {}
    away = enrichment.get("awayScheduleFlags") or {}
    adjustment = 0.0

    if home.get("backToBack"):
        adjustment -= 0.12
    if away.get("backToBack"):
        adjustment += 0.12
    if home.get("threeInFour"):
        adjustment -= 0.08
    if away.get("threeInFour"):
        adjustment += 0.08

    return max(-0.35, min(0.35, adjustment))


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def fetch_rolling_schedule_games(
    league: str,
    date_value: str,
    *,
    lookback_days: int = 7,
    current_games: list[dict[str, Any]] | None = None,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
) -> list[dict[str, Any]]:
    """Fetch prior scoreboard days plus the current slate for rest/B2B context."""
    cache_key = (league, date_value, lookback_days)
    if cache_key in _ROLLING_CACHE:
        return _ROLLING_CACHE[cache_key]

    merged: dict[str, dict[str, Any]] = {}
    anchor = _iso_date(date_value)

    for offset in range(-lookback_days, 1):
        day = (anchor + timedelta(days=offset)).isoformat()
        try:
            scoreboard = fetch_scoreboard(
                league,
                day,
                retries=retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            )
            for game in parse_scoreboard(scoreboard, league=league):
                event_id = str(game.get("eventId") or "")
                if event_id:
                    merged[event_id] = game
        except Exception:
            continue

    for game in current_games or []:
        event_id = str(game.get("eventId") or "")
        if event_id:
            merged[event_id] = game

    games = sorted(merged.values(), key=lambda item: item.get("startDate") or "")
    _ROLLING_CACHE[cache_key] = games
    return games


def clear_rolling_schedule_cache() -> None:
    _ROLLING_CACHE.clear()
