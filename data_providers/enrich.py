"""Merge external provider data into game enrichment."""

from __future__ import annotations

import time
from typing import Any

from data_providers.derived import (
    compute_rest_days,
    merge_team_profile,
    parse_weather_impact,
    series_win_pct,
)
from data_providers.league_metrics import enrich_league_metrics
from data_providers.mlb_pitcher import enrich_mlb_pitching_context
from data_providers.schedule_advanced import compute_schedule_flags
from data_providers.espn_advanced import (
    fetch_espn_standings,
    fetch_espn_team_directory,
    fetch_espn_team_statistics,
    lookup_espn_standings,
    resolve_team_id,
)
from data_providers.mlb_official import fetch_mlb_standings, lookup_mlb_official
from mlb_predictions import parse_record, win_pct_from_record


def _form_pct_from_enrichment(enrichment: dict[str, Any], side: str) -> float | None:
    form = enrichment.get(f"{side}LastFive") or {}
    record = form.get("record")
    if not record:
        return None
    return win_pct_from_record(record, default=-1.0) if parse_record(record) else None


def enrich_games_with_providers(
    games: list[dict[str, Any]],
    *,
    league: str,
    schedule_context_games: list[dict[str, Any]] | None = None,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
    request_delay: float = 0.04,
) -> list[dict[str, Any]]:
    if not games:
        return games

    context_games = schedule_context_games if schedule_context_games is not None else games

    try:
        team_directory = fetch_espn_team_directory(league, verify_ssl=verify_ssl)
    except Exception:
        team_directory = {}

    try:
        espn_standings = fetch_espn_standings(league, verify_ssl=verify_ssl)
    except Exception:
        espn_standings = {}

    mlb_standings: dict[str, dict[str, Any]] = {}
    if league == "mlb":
        try:
            mlb_standings = fetch_mlb_standings(verify_ssl=verify_ssl)
        except Exception:
            mlb_standings = {}

    team_stats_cache: dict[str, dict[str, Any]] = {}
    unique_teams = {
        name
        for game in games
        for name in (game.get("homeTeam"), game.get("awayTeam"))
        if name
    }

    for index, team_name in enumerate(sorted(unique_teams)):
        team_id = resolve_team_id(team_name, team_directory)
        if not team_id:
            continue
        try:
            team_stats_cache[team_name] = fetch_espn_team_statistics(
                league,
                team_id,
                verify_ssl=verify_ssl,
            )
        except Exception:
            team_stats_cache[team_name] = {}
        if index + 1 < len(unique_teams):
            time.sleep(request_delay)

    for game in games:
        enrichment = game.setdefault("enrichment", {})
        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")

        home_form = _form_pct_from_enrichment(enrichment, "home")
        away_form = _form_pct_from_enrichment(enrichment, "away")

        home_profile = merge_team_profile(
            league=league,
            espn_stats=team_stats_cache.get(home_team or ""),
            espn_standings=lookup_espn_standings(home_team, espn_standings),
            mlb_official=lookup_mlb_official(home_team, mlb_standings),
            form_pct=home_form,
        )
        away_profile = merge_team_profile(
            league=league,
            espn_stats=team_stats_cache.get(away_team or ""),
            espn_standings=lookup_espn_standings(away_team, espn_standings),
            mlb_official=lookup_mlb_official(away_team, mlb_standings),
            form_pct=away_form,
        )

        home_rest = compute_rest_days(context_games, home_team, game.get("startDate"))
        away_rest = compute_rest_days(context_games, away_team, game.get("startDate"))
        weather = parse_weather_impact(enrichment.get("weather"))
        series = enrichment.get("seasonSeries")
        home_h2h = series_win_pct(series, home_team)
        away_h2h = series_win_pct(series, away_team)

        enrichment["homeAdvanced"] = home_profile
        enrichment["awayAdvanced"] = away_profile
        enrichment["restDays"] = {"home": home_rest, "away": away_rest}
        enrichment["weatherImpact"] = weather
        enrichment["headToHead"] = {
            "homeSeriesWinPct": home_h2h,
            "awaySeriesWinPct": away_h2h,
            "summary": (series or {}).get("summary"),
            "seriesScore": (series or {}).get("seriesScore"),
        }

        enrichment["homeScheduleFlags"] = compute_schedule_flags(context_games, home_team, game.get("startDate"))
        enrichment["awayScheduleFlags"] = compute_schedule_flags(context_games, away_team, game.get("startDate"))
        enrichment["leagueMetrics"] = enrich_league_metrics(
            game,
            league=league,
            home_profile=home_profile,
            away_profile=away_profile,
        )

        if league == "mlb":
            enrichment["mlbPitching"] = enrich_mlb_pitching_context(game, verify_ssl=verify_ssl)

        provider_sources = sorted(
            {
                *(home_profile.get("sources") or []),
                *(away_profile.get("sources") or []),
                "Derived schedule metrics",
            }
        )
        enrichment["sources"] = sorted(set((enrichment.get("sources") or []) + provider_sources))

        if home_rest is not None and away_rest is not None and abs(home_rest - away_rest) >= 1:
            enrichment.setdefault("notes", []).append(
                f"Rest edge: {home_team} ({home_rest}d) vs {away_team} ({away_rest}d)."
            )

    return games
