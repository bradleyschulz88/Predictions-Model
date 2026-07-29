"""Merge external provider data into game enrichment."""

from __future__ import annotations

import time
from typing import Any

from data_providers.injury_severity import team_injury_severity
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
from shared_utils import parse_record, win_pct_from_record


def _form_pct_from_enrichment(
    enrichment: dict[str, Any], side: str, league: str | None = None
) -> float | None:
    """Last-five win rate, or None. A 0-0 record must not leak -1.0 downstream."""
    form = enrichment.get(f"{side}LastFive") or {}
    record = form.get("record")
    if not record or not parse_record(record):
        return None
    pct = win_pct_from_record(record, default=-1.0, league=league)
    return None if pct < 0 else pct


_series_seen = {"games": 0, "withSeries": 0, "withScore": 0, "resolved": 0, "sample": None}


def _note_series_shape(series, home_pct, away_pct) -> None:
    """Count what the season-series block actually contained, and say so once."""
    _series_seen["games"] += 1
    if series:
        _series_seen["withSeries"] += 1
        if (series or {}).get("seriesScore"):
            _series_seen["withScore"] += 1
        if _series_seen["sample"] is None:
            _series_seen["sample"] = {
                "summary": (series or {}).get("summary"),
                "seriesScore": (series or {}).get("seriesScore"),
            }
    if home_pct is not None or away_pct is not None:
        _series_seen["resolved"] += 1

    if _series_seen["games"] == SERIES_DIAGNOSTIC_AFTER:
        print(
            f"::warning title=Head-to-head coverage::of {_series_seen['games']} games, "
            f"{_series_seen['withSeries']} carried a season series, "
            f"{_series_seen['withScore']} carried a score, "
            f"{_series_seen['resolved']} resolved to a win share. "
            f"sample={_series_seen['sample']!r}"
        )


# Enough games to be representative without waiting for a whole slate.
SERIES_DIAGNOSTIC_AFTER = 25


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

        home_form = _form_pct_from_enrichment(enrichment, "home", league)
        away_form = _form_pct_from_enrichment(enrichment, "away", league)

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
        # The abbreviation is what ESPN actually writes in the series summary
        # ("TB wins series 5-1"), so without it nothing ever matched.
        home_h2h = series_win_pct(series, home_team, game.get("homeAbbr"))
        away_h2h = series_win_pct(series, away_team, game.get("awayAbbr"))

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
        # A real build resolved h2h on 22 of 120 games and every one was 0.0,
        # meaning only the tied case ever landed. That points at the season
        # series being absent rather than at the parsing, but it cannot be told
        # apart from here. Report what actually arrived, once, so the next build
        # settles it instead of another round of guessing.
        _note_series_shape(series, home_h2h, away_h2h)

        # Weighted cost of who is unavailable, as an ablation candidate. The
        # LLM importance step is off unless NVIDIA_API_KEY is set; without it
        # this is still a better read than counting absences.
        for side, team_name in (("home", home_team), ("away", away_team)):
            enrichment[f"{side}InjurySeverity"] = team_injury_severity(
                enrichment.get(f"{side}MajorInjuries"),
                league=league,
                team=team_name,
            )

        enrichment["homeScheduleFlags"] = compute_schedule_flags(context_games, home_team, game.get("startDate"))
        enrichment["awayScheduleFlags"] = compute_schedule_flags(context_games, away_team, game.get("startDate"))
        enrichment["leagueMetrics"] = enrich_league_metrics(
            game,
            league=league,
            home_profile=home_profile,
            away_profile=away_profile,
        )

        if league == "mlb":
            # Every other provider call here is guarded; this one was not, so a
            # single MLB Stats API outage failed the whole build instead of the
            # pitching enrichment alone. Downstream already treats an empty
            # context as "no pitching data".
            try:
                enrichment["mlbPitching"] = enrich_mlb_pitching_context(game, verify_ssl=verify_ssl)
            except Exception:
                enrichment["mlbPitching"] = {}

            # Recent relief workload, home minus away. Guarded the same way and
            # for the same reason: this is a candidate feature decorating a
            # prediction that is already made, so a Stats API outage must cost
            # this number and nothing else.
            try:
                from data_providers.bullpen import bullpen_edge
                from data_providers.mlb_pitcher import _resolve_team_id

                enrichment["bullpenDiff"] = bullpen_edge(
                    _resolve_team_id(home_team, verify_ssl=verify_ssl),
                    _resolve_team_id(away_team, verify_ssl=verify_ssl),
                    verify_ssl=verify_ssl,
                )
            except Exception:
                enrichment["bullpenDiff"] = None

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
