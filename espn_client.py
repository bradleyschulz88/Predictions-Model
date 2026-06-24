"""Fetch schedules from ESPN's public scoreboard API."""

from __future__ import annotations

import json
from typing import Any

from game_status import normalize_espn_status
from sbr_client import SBRClientError, get_text
from sports_config import LeagueConfig, get_league

ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports"


class ESPNClientError(SBRClientError):
    """Error while fetching or parsing ESPN schedule data."""


def iso_to_espn_date(iso_date: str) -> str:
    return iso_date.replace("-", "")


def build_scoreboard_url(league: LeagueConfig | str, date: str | None = None) -> str:
    league_config = get_league(league) if isinstance(league, str) else league
    url = f"{ESPN_API_BASE}/{league_config.espn_path}/scoreboard"
    if date:
        return f"{url}?dates={iso_to_espn_date(date)}"
    return url


def build_summary_url(league: LeagueConfig | str, event_id: str | int) -> str:
    league_config = get_league(league) if isinstance(league, str) else league
    return f"{ESPN_API_BASE}/{league_config.espn_path}/summary?event={event_id}"


def fetch_scoreboard(
    league: LeagueConfig | str,
    date: str | None = None,
    *,
    retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    url = build_scoreboard_url(league, date)
    text = get_text(url, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ESPNClientError(f"Invalid JSON from ESPN scoreboard: {exc}") from exc
    if not isinstance(data, dict):
        raise ESPNClientError("ESPN scoreboard response is not an object")
    return data


def _competitor(competitors: list[dict[str, Any]], home_away: str) -> dict[str, Any]:
    for competitor in competitors:
        if competitor.get("homeAway") == home_away:
            return competitor
    return {}


def _format_broadcasts(broadcasts: list[dict[str, Any]] | None) -> str | None:
    if not broadcasts:
        return None
    names: list[str] = []
    for item in broadcasts:
        for name in item.get("names") or []:
            if name and name not in names:
                names.append(name)
    return ", ".join(names) if names else None


def _record_by_type(records: list[dict[str, Any]] | None, *type_names: str) -> str | None:
    for type_name in type_names:
        for record in records or []:
            if record.get("type") == type_name:
                return record.get("summary")
    if records:
        return records[0].get("summary")
    return None


def _parse_probable(competitor: dict[str, Any]) -> dict[str, Any] | None:
    for item in competitor.get("probables") or []:
        if item.get("name") != "probableStartingPitcher":
            continue
        athlete = item.get("athlete") or {}
        era = None
        for stat in item.get("statistics") or []:
            if stat.get("abbreviation") == "ERA":
                try:
                    era = float(stat.get("displayValue"))
                except (TypeError, ValueError):
                    era = None
        player_id = item.get("playerId") or athlete.get("id")
        parsed_id = None
        if player_id is not None:
            try:
                parsed_id = int(player_id)
            except (TypeError, ValueError):
                parsed_id = None
        return {
            "name": athlete.get("displayName"),
            "playerId": parsed_id,
            "era": era,
            "record": item.get("record"),
        }
    return None


def parse_scoreboard(scoreboard: dict[str, Any], *, league: LeagueConfig | str) -> list[dict[str, Any]]:
    league_config = get_league(league) if isinstance(league, str) else league
    events = scoreboard.get("events")
    if not isinstance(events, list):
        raise ESPNClientError("ESPN scoreboard missing events list")

    games: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        competition = competitions[0]
        if not isinstance(competition, dict):
            continue

        away = _competitor(competition.get("competitors") or [], "away")
        home = _competitor(competition.get("competitors") or [], "home")
        away_team = (away.get("team") or {}).get("displayName")
        home_team = (home.get("team") or {}).get("displayName")
        venue = (competition.get("venue") or {}).get("fullName")

        away_records = away.get("records")
        home_records = home.get("records")
        away_record = _record_by_type(away_records, "total", "standingsoverall")
        home_record = _record_by_type(home_records, "total", "standingsoverall")
        away_road_record = _record_by_type(away_records, "road", "away")
        home_home_record = _record_by_type(home_records, "home")

        status = (event.get("status") or {}).get("type") or {}
        status_flags = normalize_espn_status(
            status,
            start_date=competition.get("date") or event.get("date"),
            attendance=competition.get("attendance"),
            notes=competition.get("notes"),
            home_score=home.get("score"),
            away_score=away.get("score"),
        )

        game: dict[str, Any] = {
            "league": league_config.id,
            "leagueLabel": league_config.label,
            "eventId": event.get("id"),
            "startDate": competition.get("date") or event.get("date"),
            "awayTeam": away_team,
            "homeTeam": home_team,
            "matchup": f"{away_team} @ {home_team}" if away_team and home_team else event.get("name"),
            "gameStatusText": status_flags["gameStatusText"],
            "gameStatusDetail": status_flags["gameStatusDetail"],
            "statusType": status_flags["statusType"],
            "venueName": venue,
            "broadcast": _format_broadcasts(competition.get("broadcasts")),
            "awayRecord": away_record,
            "homeRecord": home_record,
            "awayRoadRecord": away_road_record,
            "homeHomeRecord": home_home_record,
            "awayScore": away.get("score"),
            "homeScore": home.get("score"),
            "isLive": status_flags["isLive"],
            "isFinal": status_flags["isFinal"],
            "isScheduled": status_flags["isScheduled"],
            "isPostponed": status_flags["isPostponed"],
            "isCanceled": status_flags["isCanceled"],
            "isSuspended": status_flags["isSuspended"],
            "isDelayed": status_flags["isDelayed"],
            "isVoided": status_flags["isVoided"],
            "isWashedOut": status_flags["isWashedOut"],
            "attendance": status_flags.get("attendance"),
            "source": "espn",
            "viewTypes": [],
            "lines": [],
        }

        if league_config.supports_pitchers:
            game["awayPitcher"] = _parse_probable(away)
            game["homePitcher"] = _parse_probable(home)

        games.append(game)

    games.sort(key=lambda item: item.get("startDate") or "")
    return games
