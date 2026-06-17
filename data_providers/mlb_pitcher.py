"""MLB Stats API — pitcher and bullpen context for matchup modeling."""

from __future__ import annotations

from typing import Any

from urllib.parse import urlencode

from data_providers.utils import fetch_json, to_float


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


def _pitcher_season_era(player_id: int, *, verify_ssl: bool = True) -> float | None:
    payload = _fetch_api(
        f"/api/v1/people/{player_id}/stats",
        {"stats": "season", "group": "pitching"},
        cache_key=f"mlb:pitcher:season:{player_id}",
        verify_ssl=verify_ssl,
    )
    for group in payload.get("stats") or []:
        for split in group.get("splits") or []:
            era = to_float((split.get("stat") or {}).get("era"))
            if era is not None:
                return era
    return None


def _team_bullpen_era(team_name: str | None, *, verify_ssl: bool = True) -> float | None:
    if not team_name:
        return None
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
    """Attach supplemental SP/bullpen ERA from MLB Stats API when ESPN data is thin."""
    context: dict[str, Any] = {"sources": []}
    for side in ("home", "away"):
        pitcher = game.get(f"{side}Pitcher") or {}
        name = pitcher.get("name")
        api_era = None
        player_id = _search_player_id(name, verify_ssl=verify_ssl)
        if player_id:
            api_era = _pitcher_season_era(player_id, verify_ssl=verify_ssl)
            if api_era is not None:
                context[f"{side}PitcherApiEra"] = api_era
                context["sources"].append("MLB Stats API pitcher")

        if pitcher.get("era") is None and api_era is not None:
            pitcher["era"] = api_era
            game[f"{side}Pitcher"] = pitcher

        team = game.get(f"{side}Team")
        bullpen_era = _team_bullpen_era(team, verify_ssl=verify_ssl)
        if bullpen_era is not None:
            context[f"{side}BullpenEra"] = bullpen_era
            if "MLB Stats API bullpen" not in context["sources"]:
                context["sources"].append("MLB Stats API bullpen")

    if context["sources"]:
        context["sources"] = sorted(set(context["sources"]))
    return context


def mlb_pitching_logit_adjustment(game: dict[str, Any], enrichment: dict[str, Any]) -> float:
    pitching = enrichment.get("mlbPitching") or {}
    adjustment = 0.0

    for side, other in (("home", "away"), ("away", "home")):
        bullpen = pitching.get(f"{side}BullpenEra")
        other_bullpen = pitching.get(f"{other}BullpenEra")
        if bullpen is not None and other_bullpen is not None:
            adjustment += (other_bullpen - bullpen) * 0.08

        api_era = pitching.get(f"{side}PitcherApiEra")
        other_api = pitching.get(f"{other}PitcherApiEra")
        if api_era is not None and other_api is not None:
            adjustment += (other_api - api_era) * 0.15

    return max(-0.5, min(0.5, adjustment))
