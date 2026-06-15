"""Derived signals from schedule and enrichment (rest, weather, power index)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def parse_weather_impact(weather: str | None) -> dict[str, Any] | None:
    if not weather:
        return None
    temp_match = re.search(r"(-?\d+)\s*°", weather)
    precip_match = re.search(r"(\d+)%\s*precip", weather, re.I)
    wind_match = re.search(r"(\d+)\s*mph\s*wind", weather, re.I)
    temp_f = float(temp_match.group(1)) if temp_match else None
    precip = float(precip_match.group(1)) if precip_match else None
    wind = float(wind_match.group(1)) if wind_match else None

    run_env = 0.0
    notes: list[str] = []
    if temp_f is not None:
        if temp_f >= 82:
            run_env += 0.04
            notes.append("warm air favors offense")
        elif temp_f <= 50:
            run_env -= 0.03
            notes.append("cold air suppresses scoring")
    if wind is not None and wind >= 12:
        run_env += 0.02
        notes.append("wind can affect fly balls")
    if precip is not None and precip >= 40:
        notes.append("rain risk")

    return {
        "temperatureF": temp_f,
        "precipitationPct": precip,
        "windMph": wind,
        "runEnvironmentAdj": run_env,
        "summary": "; ".join(notes) if notes else None,
    }


def series_win_pct(series: dict[str, Any] | None, team_name: str | None) -> float | None:
    if not series or not team_name:
        return None
    summary = series.get("summary") or ""
    score = series.get("seriesScore") or ""
    if team_name.lower() not in summary.lower() and team_name.split()[-1].lower() not in summary.lower():
        return None
    match = re.search(r"(\d+)-(\d+)", score)
    if not match:
        return None
    left, right = int(match.group(1)), int(match.group(2))
    total = left + right
    if not total:
        return None
    if summary.lower().index(team_name.split()[-1].lower()) < len(summary) / 2:
        return left / total
    return right / total


def compute_rest_days(games: list[dict[str, Any]], team_name: str | None, current_start: str | None) -> int | None:
    if not team_name or not current_start:
        return None
    try:
        current_dt = datetime.fromisoformat(current_start.replace("Z", "+00:00"))
    except ValueError:
        return None

    previous: datetime | None = None
    for game in games:
        start = game.get("startDate")
        if not start:
            continue
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            continue
        if start_dt >= current_dt:
            continue
        if team_name not in {game.get("homeTeam"), game.get("awayTeam")}:
            continue
        if previous is None or start_dt > previous:
            previous = start_dt

    if previous is None:
        return None
    delta = current_dt - previous
    return max(0, delta.days)


def compute_power_rating(
    *,
    league: str,
    win_pct: float | None,
    run_diff_per_game: float | None = None,
    goals_for_per_game: float | None = None,
    goals_against_per_game: float | None = None,
    form_pct: float | None = None,
    batting_ops_proxy: float | None = None,
    era: float | None = None,
) -> float | None:
    parts: list[tuple[float, float]] = []
    if win_pct is not None:
        parts.append((win_pct, 0.35))
    if form_pct is not None and form_pct >= 0:
        parts.append((form_pct, 0.20))
    if league == "mlb":
        if run_diff_per_game is not None:
            normalized = max(0.0, min(1.0, (run_diff_per_game + 2.0) / 4.0))
            parts.append((normalized, 0.25))
        if batting_ops_proxy is not None:
            parts.append((max(0.0, min(1.0, (batting_ops_proxy - 0.650) / 0.150)), 0.10))
        if era is not None:
            parts.append((max(0.0, min(1.0, (5.5 - era) / 2.5)), 0.10))
    elif league in {"epl", "worldcup"}:
        if goals_for_per_game is not None and goals_against_per_game is not None:
            attack = max(0.0, min(1.0, goals_for_per_game / 3.0))
            defense = max(0.0, min(1.0, (3.0 - goals_against_per_game) / 2.0))
            parts.append((attack, 0.20))
            parts.append((defense, 0.15))
    elif league in {"nba", "nfl", "afl"}:
        if goals_for_per_game is not None:
            parts.append((max(0.0, min(1.0, goals_for_per_game / 130.0)), 0.15))

    if not parts:
        return None
    weight_total = sum(weight for _, weight in parts)
    return sum(value * weight for value, weight in parts) / weight_total


def merge_team_profile(
    *,
    league: str,
    espn_stats: dict[str, Any] | None,
    espn_standings: dict[str, Any] | None,
    mlb_official: dict[str, Any] | None,
    form_pct: float | None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {"sources": []}

    if espn_standings:
        profile.update(
            {
                "wins": espn_standings.get("wins"),
                "losses": espn_standings.get("losses"),
                "points": espn_standings.get("points"),
                "goalDifference": espn_standings.get("goalDifference"),
                "pointsPerGame": espn_standings.get("pointsPerGame"),
                "goalsAgainstPerGame": espn_standings.get("goalsAgainstPerGame"),
            }
        )
        profile["sources"].append("ESPN standings")

    if mlb_official:
        profile.update(
            {
                "runDifferential": mlb_official.get("runDifferential"),
                "runsPerGame": mlb_official.get("runsPerGame"),
                "runsAllowedPerGame": mlb_official.get("runsAllowedPerGame"),
                "streakCode": mlb_official.get("streakCode"),
                "streakNumber": mlb_official.get("streakNumber"),
                "streakType": mlb_official.get("streakType"),
            }
        )
        profile["sources"].append("MLB.com")

    if espn_stats:
        profile.update(
            {
                "battingAvg": espn_stats.get("battingAvg"),
                "onBasePct": espn_stats.get("onBasePct"),
                "sluggingPct": espn_stats.get("sluggingPct"),
                "era": espn_stats.get("era"),
                "runsScored": espn_stats.get("runsScored"),
                "runsAllowed": espn_stats.get("runsAllowed"),
            }
        )
        profile["sources"].append("ESPN team stats")

    win_pct = None
    if espn_standings and espn_standings.get("winPct") is not None:
        win_pct = espn_standings["winPct"]
    elif mlb_official and mlb_official.get("winPct") is not None:
        win_pct = mlb_official["winPct"]

    run_diff_pg = None
    if mlb_official and mlb_official.get("runDifferential") is not None:
        games = mlb_official.get("gamesPlayed") or 1
        run_diff_pg = mlb_official["runDifferential"] / games

    ops_proxy = None
    if espn_stats and espn_stats.get("onBasePct") is not None and espn_stats.get("sluggingPct") is not None:
        ops_proxy = espn_stats["onBasePct"] + espn_stats["sluggingPct"]

    profile["powerRating"] = compute_power_rating(
        league=league,
        win_pct=win_pct,
        run_diff_per_game=run_diff_pg,
        goals_for_per_game=espn_standings.get("pointsPerGame") if espn_standings else None,
        goals_against_per_game=espn_standings.get("goalsAgainstPerGame") if espn_standings else None,
        form_pct=form_pct,
        batting_ops_proxy=ops_proxy,
        era=espn_stats.get("era") if espn_stats else None,
    )
    profile["opsProxy"] = ops_proxy
    profile["winPct"] = win_pct
    profile["sources"] = sorted(set(profile["sources"]))
    return profile
