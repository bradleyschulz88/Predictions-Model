"""MLB Stats API — pitcher and bullpen context for matchup modeling."""

from __future__ import annotations

from typing import Any

from urllib.parse import urlencode

from data_providers.utils import fetch_json, to_float

_TEAM_ID_CACHE: dict[str, int | None] = {}


def _fetch_api(path: str, params: dict[str, str] | None = None, *, cache_key: str, verify_ssl: bool = True) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    return fetch_json(f"https://statsapi.mlb.com{path}{query}", cache_key=cache_key, verify_ssl=verify_ssl)


def _search_player_id(
    name: str | None,
    *,
    team_name: str | None = None,
    verify_ssl: bool = True,
) -> int | None:
    if not name:
        return None
    payload = _fetch_api(
        "/api/v1/people/search",
        {"names": name},
        cache_key=f"mlb:playersearch:{name.lower()}:{team_name or ''}",
        verify_ssl=verify_ssl,
    )
    people = payload.get("people") or []
    if not people:
        return None

    if team_name:
        for person in people:
            current_team = (person.get("currentTeam") or {}).get("name") or ""
            if current_team and (current_team in team_name or team_name in current_team):
                return int(person["id"])

    for person in people:
        if person.get("fullName"):
            return int(person["id"])
    return None


def _resolve_pitcher_id(pitcher: dict[str, Any], team_name: str | None, *, verify_ssl: bool = True) -> int | None:
    stored = pitcher.get("mlbId")
    if stored is not None:
        try:
            return int(stored)
        except (TypeError, ValueError):
            pass
    return _search_player_id(pitcher.get("name"), team_name=team_name, verify_ssl=verify_ssl)


def _team_reliever_era(team_id: int, *, verify_ssl: bool = True) -> float | None:
    """IP-weighted reliever ERA from MLB Stats API."""
    payload = _fetch_api(
        "/api/v1/stats",
        {
            "stats": "season",
            "group": "pitching",
            "playerPool": "all",
            "teamId": str(team_id),
            "position": "R",
        },
        cache_key=f"mlb:bullpen:relievers:{team_id}",
        verify_ssl=verify_ssl,
    )
    total_ip = 0.0
    total_er = 0.0
    for group in payload.get("stats") or []:
        for split in group.get("splits") or []:
            stat = split.get("stat") or {}
            ip = to_float(stat.get("inningsPitched"))
            er = to_float(stat.get("earnedRuns"))
            if ip is None or er is None or ip <= 0:
                continue
            total_ip += ip
            total_er += er
    if total_ip <= 0:
        return None
    return round(total_er / total_ip * 9.0, 2)


def _team_bullpen_era(team_name: str | None, *, verify_ssl: bool = True) -> float | None:
    team_id = _resolve_team_id(team_name, verify_ssl=verify_ssl)
    if not team_id:
        return None
    bullpen_era = _team_reliever_era(team_id, verify_ssl=verify_ssl)
    if bullpen_era is not None:
        return bullpen_era
    return _team_pitching_era(team_name, verify_ssl=verify_ssl)


def _pitcher_season_stats(player_id: int, *, verify_ssl: bool = True) -> dict[str, float | None]:
    payload = _fetch_api(
        f"/api/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching"},
        cache_key=f"mlb:pitcher:season:{player_id}",
        verify_ssl=verify_ssl,
    )
    era = None
    fip = None
    for group in payload.get("stats") or []:
        for split in group.get("splits") or []:
            stat = split.get("stat") or {}
            era = era if era is not None else to_float(stat.get("era"))
            fip = fip if fip is not None else to_float(stat.get("fip"))
    return {"era": era, "fip": fip}


def _pitcher_recent_start_era(player_id: int, *, starts: int = 3, verify_ssl: bool = True) -> float | None:
    payload = _fetch_api(
        f"/api/v1/people/{player_id}/stats",
        {"stats": "gameLog", "group": "pitching"},
        cache_key=f"mlb:pitcher:gamelog:{player_id}",
        verify_ssl=verify_ssl,
    )
    innings: list[float] = []
    earned: list[float] = []
    for group in payload.get("stats") or []:
        for split in group.get("splits") or []:
            stat = split.get("stat") or {}
            ip = to_float(stat.get("inningsPitched"))
            er = to_float(stat.get("earnedRuns"))
            if ip is None or er is None or ip <= 0:
                continue
            innings.append(ip)
            earned.append(er)
            if len(innings) >= starts:
                break
        if len(innings) >= starts:
            break
    if not innings:
        return None
    total_ip = sum(innings)
    if total_ip <= 0:
        return None
    return round(sum(earned) / total_ip * 9.0, 2)


def _resolve_team_id(team_name: str | None, *, verify_ssl: bool = True) -> int | None:
    if not team_name:
        return None
    if team_name in _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE[team_name]

    teams_payload = _fetch_api(
        "/api/v1/teams",
        {"sportIds": "1"},
        cache_key="mlb:teams:active",
        verify_ssl=verify_ssl,
    )
    team_id = None
    for team in teams_payload.get("teams") or []:
        if team.get("name") == team_name or team.get("teamName") in team_name:
            team_id = team.get("id")
            break
    _TEAM_ID_CACHE[team_name] = int(team_id) if team_id is not None else None
    return _TEAM_ID_CACHE[team_name]


def _team_pitching_era(team_name: str | None, *, verify_ssl: bool = True) -> float | None:
    team_id = _resolve_team_id(team_name, verify_ssl=verify_ssl)
    if not team_id:
        return None

    stats_payload = _fetch_api(
        f"/api/v1/teams/{team_id}/stats",
        {"stats": "season", "group": "pitching"},
        cache_key=f"mlb:team:pitching:{team_id}",
        verify_ssl=verify_ssl,
    )
    for group in stats_payload.get("stats") or []:
        for split in group.get("splits") or []:
            era = to_float((split.get("stat") or {}).get("era"))
            if era is not None:
                return era
    return None


def enrich_mlb_pitching_context(game: dict[str, Any], *, verify_ssl: bool = True) -> dict[str, Any]:
    """Attach supplemental SP/team pitching ERA from MLB Stats API when ESPN data is thin."""
    context: dict[str, Any] = {"sources": []}
    for side in ("home", "away"):
        pitcher = game.get(f"{side}Pitcher") or {}
        team = game.get(f"{side}Team")
        player_id = _resolve_pitcher_id(pitcher, team, verify_ssl=verify_ssl)
        if player_id:
            if pitcher.get("playerId") is not None:
                pitcher["espnPlayerId"] = pitcher.get("playerId")
            pitcher["mlbId"] = player_id
            game[f"{side}Pitcher"] = pitcher
            season = _pitcher_season_stats(player_id, verify_ssl=verify_ssl)
            api_era = season.get("era")
            api_fip = season.get("fip")
            recent_era = _pitcher_recent_start_era(player_id, verify_ssl=verify_ssl)
            if api_era is not None:
                context[f"{side}PitcherApiEra"] = api_era
                context["sources"].append("MLB Stats API pitcher")
            if api_fip is not None:
                context[f"{side}PitcherFip"] = api_fip
            if recent_era is not None:
                context[f"{side}PitcherRecentEra"] = recent_era

            if pitcher.get("era") is None and api_era is not None:
                pitcher["era"] = api_era
            if api_fip is not None:
                pitcher["fip"] = api_fip
            game[f"{side}Pitcher"] = pitcher

        team = game.get(f"{side}Team")
        team_era = _team_pitching_era(team, verify_ssl=verify_ssl)
        bullpen_era = _team_bullpen_era(team, verify_ssl=verify_ssl)
        if team_era is not None:
            context[f"{side}TeamPitchingEra"] = team_era
            if "MLB Stats API team pitching" not in context["sources"]:
                context["sources"].append("MLB Stats API team pitching")
        if bullpen_era is not None:
            context[f"{side}BullpenEra"] = bullpen_era
            if "MLB Stats API bullpen" not in context["sources"]:
                context["sources"].append("MLB Stats API bullpen")

    if context["sources"]:
        context["sources"] = sorted(set(context["sources"]))
    return context


def mlb_pitching_logit_adjustment(game: dict[str, Any], enrichment: dict[str, Any]) -> float:
    """Bullpen-only adjustment; starting pitching handled separately in predict_game."""
    pitching = enrichment.get("mlbPitching") or {}
    home_bp = pitching.get("homeBullpenEra")
    away_bp = pitching.get("awayBullpenEra")
    if home_bp is None or away_bp is None:
        return 0.0
    adjustment = (away_bp - home_bp) * 0.06
    return max(-0.35, min(0.35, adjustment))
