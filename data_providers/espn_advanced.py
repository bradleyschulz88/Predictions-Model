"""ESPN team statistics and standings (all supported leagues)."""

from __future__ import annotations

from typing import Any

from data_providers.utils import best_team_match, fetch_json, normalize_team_name, stat_map, to_float
from sports_config import LeagueConfig, get_league


def _flatten_espn_teams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    teams: list[dict[str, Any]] = []
    for sport in payload.get("sports") or []:
        for league in sport.get("leagues") or []:
            for entry in league.get("teams") or []:
                team = entry.get("team") or entry
                if team.get("id"):
                    teams.append(team)
    return teams


def fetch_espn_team_directory(league: str, *, verify_ssl: bool = True) -> dict[str, dict[str, Any]]:
    league_config = get_league(league)
    cache_key = f"espn:teams:{league}"
    payload = fetch_json(
        f"https://site.api.espn.com/apis/site/v2/sports/{league_config.espn_path}/teams",
        cache_key=cache_key,
        verify_ssl=verify_ssl,
    )
    directory: dict[str, dict[str, Any]] = {}
    for team in _flatten_espn_teams(payload):
        display = team.get("displayName") or team.get("name")
        if not display:
            continue
        directory[normalize_team_name(display)] = {
            "id": str(team.get("id")),
            "displayName": display,
            "abbreviation": team.get("abbreviation"),
            "slug": team.get("slug"),
        }
    return directory


def fetch_espn_team_statistics(
    league: str,
    team_id: str,
    *,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    league_config = get_league(league)
    cache_key = f"espn:teamstats:{league}:{team_id}"
    payload = fetch_json(
        f"https://site.api.espn.com/apis/site/v2/sports/{league_config.espn_path}/teams/{team_id}/statistics",
        cache_key=cache_key,
        verify_ssl=verify_ssl,
    )
    stats_root = ((payload.get("results") or {}).get("stats") or {})
    categories = stats_root.get("categories") or []
    batting: dict[str, Any] = {}
    pitching: dict[str, Any] = {}
    general: dict[str, Any] = {}
    for category in categories:
        name = (category.get("name") or "").lower()
        values = stat_map(category.get("stats"))
        if name == "batting":
            batting = values
        elif name == "pitching":
            pitching = values
        else:
            general.update(values)

    return {
        "battingAvg": to_float(batting.get("avg")),
        "onBasePct": to_float(batting.get("onBasePct")),
        "sluggingPct": to_float(batting.get("slugAvg")),
        "runsScored": to_float(batting.get("runs")),
        "homeRuns": to_float(batting.get("homeRuns")),
        "era": to_float(pitching.get("ERA")),
        "runsAllowed": to_float(pitching.get("runs")),
        "wins": to_float(pitching.get("wins") or general.get("wins")),
        "losses": to_float(pitching.get("losses") or general.get("losses")),
        "pointsFor": to_float(general.get("pointsFor") or batting.get("points")),
        "pointsAgainst": to_float(general.get("pointsAgainst")),
        "source": "ESPN",
    }


def fetch_espn_standings(league: str, *, verify_ssl: bool = True) -> dict[str, dict[str, Any]]:
    league_config = get_league(league)
    cache_key = f"espn:standings:{league}"
    payload = fetch_json(
        f"https://site.api.espn.com/apis/v2/sports/{league_config.espn_path}/standings",
        cache_key=cache_key,
        verify_ssl=verify_ssl,
    )

    teams: dict[str, dict[str, Any]] = {}
    for child in payload.get("children") or []:
        standings = child.get("standings") or {}
        for entry in standings.get("entries") or []:
            team_info = entry.get("team") or {}
            display = team_info.get("displayName") or team_info.get("name")
            if not display:
                continue
            stats = stat_map(entry.get("stats"))
            games = to_float(stats.get("gamesPlayed")) or 0.0
            wins = to_float(stats.get("wins")) or 0.0
            losses = to_float(stats.get("losses")) or 0.0
            ties = to_float(stats.get("ties")) or 0.0
            points = to_float(stats.get("points"))
            points_for = to_float(stats.get("pointsFor"))
            points_against = to_float(stats.get("pointsAgainst"))
            teams[normalize_team_name(display)] = {
                "teamName": display,
                "wins": int(wins),
                "losses": int(losses),
                "ties": int(ties),
                "gamesPlayed": int(games),
                "points": points,
                "pointsFor": points_for,
                "pointsAgainst": points_against,
                "goalDifference": (points_for - points_against) if points_for is not None and points_against is not None else None,
                "winPct": wins / games if games else None,
                "pointsPerGame": points_for / games if points_for is not None and games else None,
                "goalsAgainstPerGame": points_against / games if points_against is not None and games else None,
                "source": "ESPN",
            }
    return teams


def resolve_team_id(team_name: str | None, directory: dict[str, dict[str, Any]]) -> str | None:
    if not team_name:
        return None
    key = best_team_match(team_name, directory)
    if not key:
        return None
    return directory[key].get("id")


def lookup_espn_standings(team_name: str | None, standings: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not team_name:
        return None
    key = best_team_match(team_name, standings)
    return standings.get(key) if key else None


def league_config_path(league_config: LeagueConfig) -> str:
    return league_config.espn_path
