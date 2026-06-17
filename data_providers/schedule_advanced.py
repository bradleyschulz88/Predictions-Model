"""Schedule-derived flags: back-to-back, three-in-four, travel proxy."""

from __future__ import annotations

from datetime import datetime
from typing import Any


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
