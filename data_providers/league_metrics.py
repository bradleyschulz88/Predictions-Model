"""Per-league advanced metrics derived from standings and ESPN stats."""

from __future__ import annotations

from typing import Any

LEAGUE_DRAW_BASE_RATES = {
    "epl": 0.25,
    "worldcup": 0.22,
}
DEFAULT_DRAW_RATE = 0.26


def enrich_league_metrics(
    game: dict[str, Any],
    *,
    league: str,
    home_profile: dict[str, Any],
    away_profile: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"league": league}

    if league in {"nba", "wnba"}:
        home_ppg = home_profile.get("pointsPerGame")
        away_ppg = away_profile.get("pointsPerGame")
        if home_ppg is not None and away_ppg is not None:
            combined_pace = home_ppg + away_ppg
            metrics["paceProxy"] = round(combined_pace, 2)
            metrics["homePaceEdge"] = round(home_ppg - away_ppg, 2)

    if league == "nfl":
        home_pf = home_profile.get("pointsPerGame")
        away_pf = away_profile.get("pointsPerGame")
        home_pa = home_profile.get("goalsAgainstPerGame")
        away_pa = away_profile.get("goalsAgainstPerGame")
        if home_pf is not None and away_pf is not None and home_pa is not None and away_pa is not None:
            home_eff = home_pf - home_pa
            away_eff = away_pf - away_pa
            metrics["homeEfficiency"] = round(home_eff, 2)
            metrics["awayEfficiency"] = round(away_eff, 2)
            metrics["efficiencyEdge"] = round(home_eff - away_eff, 2)

    if league in {"epl", "worldcup"}:
        home_gd = home_profile.get("goalDifference")
        away_gd = away_profile.get("goalDifference")
        if home_gd is not None and away_gd is not None:
            metrics["goalDiffEdge"] = home_gd - away_gd
        metrics["drawBaseRate"] = LEAGUE_DRAW_BASE_RATES.get(league, DEFAULT_DRAW_RATE)

    return metrics


def league_metrics_logit_adjustment(enrichment: dict[str, Any], league: str) -> float:
    metrics = enrichment.get("leagueMetrics") or {}
    adjustment = 0.0

    if league in {"nba", "wnba"}:
        edge = metrics.get("homePaceEdge")
        if edge is not None:
            adjustment += edge * 0.02

    if league == "nfl":
        edge = metrics.get("efficiencyEdge")
        if edge is not None:
            adjustment += edge * 0.04

    if league in {"epl", "worldcup"}:
        edge = metrics.get("goalDiffEdge")
        if edge is not None:
            adjustment += max(-0.4, min(0.4, edge * 0.01))

    return max(-0.45, min(0.45, adjustment))


def soccer_draw_probability(
    *,
    league: str,
    home_true: float,
    away_true: float,
    enrichment: dict[str, Any],
) -> float:
    metrics = enrichment.get("leagueMetrics") or {}
    base = metrics.get("drawBaseRate") or LEAGUE_DRAW_BASE_RATES.get(league, DEFAULT_DRAW_RATE)
    parity = 1.0 - abs(home_true - away_true)
    goal_edge = metrics.get("goalDiffEdge")
    parity_boost = 0.05 if goal_edge is not None and abs(goal_edge) <= 3 else 0.0
    return max(0.08, min(0.32, base * (0.75 + 0.5 * parity) + parity_boost))
