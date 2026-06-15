"""MLB.com Stats API — official standings, run differential, streaks."""

from __future__ import annotations

from datetime import date
from typing import Any

from data_providers.utils import best_team_match, fetch_json, normalize_team_name, to_float


def fetch_mlb_standings(*, season: int | None = None, verify_ssl: bool = True) -> dict[str, dict[str, Any]]:
    season = season or date.today().year
    cache_key = f"mlb_official:standings:{season}"
    data = fetch_json(
        f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason",
        cache_key=cache_key,
        verify_ssl=verify_ssl,
    )

    teams: dict[str, dict[str, Any]] = {}
    for record_group in data.get("records") or []:
        for team_record in record_group.get("teamRecords") or []:
            team_info = team_record.get("team") or {}
            name = team_info.get("name") or team_info.get("teamName")
            if not name:
                continue
            streak = team_record.get("streak") or {}
            games = to_float(team_record.get("gamesPlayed")) or 0.0
            wins = to_float(team_record.get("wins")) or 0.0
            losses = to_float(team_record.get("losses")) or 0.0
            run_diff = to_float(team_record.get("runDifferential"))
            runs_scored = to_float(team_record.get("runsScored"))
            runs_allowed = to_float(team_record.get("runsAllowed"))
            teams[normalize_team_name(name)] = {
                "teamName": name,
                "wins": int(wins),
                "losses": int(losses),
                "gamesPlayed": int(games),
                "winPct": wins / (wins + losses) if wins + losses else None,
                "runDifferential": run_diff,
                "runsScored": runs_scored,
                "runsAllowed": runs_allowed,
                "runsPerGame": runs_scored / games if runs_scored is not None and games else None,
                "runsAllowedPerGame": runs_allowed / games if runs_allowed is not None and games else None,
                "streakCode": streak.get("streakCode"),
                "streakNumber": to_float(streak.get("streakNumber")),
                "streakType": streak.get("streakType"),
                "divisionRank": team_record.get("divisionRank"),
                "source": "MLB.com",
            }
    return teams


def lookup_mlb_official(team_name: str | None, standings: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not team_name or not standings:
        return None
    key = best_team_match(team_name, standings)
    return standings.get(key) if key else None
