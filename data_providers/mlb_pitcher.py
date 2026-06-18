"""MLB Stats API — pitcher and bullpen context for matchup modeling."""

from __future__ import annotations

from typing import Any

from urllib.parse import urlencode

from data_providers.utils import fetch_json, to_float

_TEAM_ID_CACHE: dict[str, int | None] = {}


def _fetch_api(path: str, params: dict[str, str] | None = None, *, cache_key: str, verify_ssl: bool = True) -> dict:
    query = f"?{urlencode(params)}" if params else ""
    return fetch_json(f"https://statsapi.mlb.com{path}{query}", cache_key=cache_key, verify_ssl=verify_ssl)


def _search_player_id(name: str | None, *, verify_ssl: bool = True) -> int | None:
    if not name:
        return None
    payload = _fetch_api(
        "/api/v1/people/search",
        {"names": name},
        cache_key=f"mlb:playersearch:{name.lower()}",
        verify_ssl=verify_ssl,
    )
    for person in payload.get("people") or []:
        if person.get("fullName"):
            return int(person["id"])
    return None


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
        name = pitcher.get("name")
        player_id = _search_player_id(name, verify_ssl=verify_ssl)
        if player_id:
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
        if team_era is not None:
            context[f"{side}TeamPitchingEra"] = team_era
            context[f"{side}BullpenEra"] = team_era
            if "MLB Stats API team pitching" not in context["sources"]:
                context["sources"].append("MLB Stats API team pitching")

    if context["sources"]:
        context["sources"] = sorted(set(context["sources"]))
    return context


def mlb_pitching_logit_adjustment(game: dict[str, Any], enrichment: dict[str, Any]) -> float:
    pitching = enrichment.get("mlbPitching") or {}
    adjustment = 0.0

    for side, other in (("home", "away"), ("away", "home")):
        team_pitching = pitching.get(f"{side}TeamPitchingEra") or pitching.get(f"{side}BullpenEra")
        other_pitching = pitching.get(f"{other}TeamPitchingEra") or pitching.get(f"{other}BullpenEra")
        if team_pitching is not None and other_pitching is not None:
            adjustment += (other_pitching - team_pitching) * 0.04

        api_era = pitching.get(f"{side}PitcherApiEra")
        other_api = pitching.get(f"{other}PitcherApiEra")
        if api_era is not None and other_api is not None:
            adjustment += (other_api - api_era) * 0.15

        api_fip = pitching.get(f"{side}PitcherFip")
        other_fip = pitching.get(f"{other}PitcherFip")
        if api_fip is not None and other_fip is not None:
            adjustment += (other_fip - api_fip) * 0.10

        recent = pitching.get(f"{side}PitcherRecentEra")
        other_recent = pitching.get(f"{other}PitcherRecentEra")
        if recent is not None and other_recent is not None:
            adjustment += (other_recent - recent) * 0.12

    return max(-0.5, min(0.5, adjustment))
