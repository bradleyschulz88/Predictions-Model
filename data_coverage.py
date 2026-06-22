"""Data coverage metrics for model inputs across built game payloads."""

from __future__ import annotations

from typing import Any

COVERAGE_FLAGS = (
    "espnPredictor",
    "lineup",
    "impliedOdds",
    "mlbPitching",
    "restData",
    "scheduleFlags",
    "advancedStats",
    "injuries",
)

PREDICTOR_COVERAGE_WARN_THRESHOLD = 20.0


def coverage_from_game(game: dict[str, Any]) -> dict[str, bool]:
    enrichment = game.get("enrichment") or {}
    home_adv = enrichment.get("homeAdvanced") or {}
    away_adv = enrichment.get("awayAdvanced") or {}
    rest_days = enrichment.get("restDays") or {}
    home_flags = enrichment.get("homeScheduleFlags") or {}
    away_flags = enrichment.get("awayScheduleFlags") or {}
    prediction = game.get("prediction") or {}
    features = prediction.get("features") or {}
    feature_coverage = features.get("dataCoverage") or {}

    if feature_coverage:
        return {flag: bool(feature_coverage.get(flag)) for flag in COVERAGE_FLAGS}

    from mlb_predictions import compute_implied_probabilities, has_moneyline_lines

    implied = compute_implied_probabilities(game.get("lines") or [])
    return {
        "espnPredictor": enrichment.get("espnPredictorHome") is not None
        and enrichment.get("espnPredictorAway") is not None,
        "lineup": bool((game.get("homeLineup") or {}).get("batters") or (game.get("awayLineup") or {}).get("batters")),
        "impliedOdds": bool(implied.get("available")) or has_moneyline_lines(game.get("lines") or []),
        "mlbPitching": bool(enrichment.get("mlbPitching")),
        "restData": rest_days.get("home") is not None and rest_days.get("away") is not None,
        "scheduleFlags": bool(home_flags or away_flags),
        "advancedStats": home_adv.get("powerRating") is not None or away_adv.get("powerRating") is not None,
        "injuries": bool(enrichment.get("homeMajorInjuries") or enrichment.get("awayMajorInjuries")),
    }


def summarize_coverage(games: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(games)
    counts = {flag: 0 for flag in COVERAGE_FLAGS}
    for game in games:
        flags = coverage_from_game(game)
        for flag, present in flags.items():
            if present:
                counts[flag] += 1

    pct = {
        flag: round(counts[flag] / total * 100, 1) if total else 0.0
        for flag in COVERAGE_FLAGS
    }
    return {
        "gameCount": total,
        "counts": counts,
        "pct": pct,
    }


def summarize_league_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_league: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        league = payload.get("league") or "unknown"
        games = payload.get("games") or []
        by_league[league] = summarize_coverage(games)
    return by_league


def coverage_warnings(
    coverage_by_league: dict[str, dict[str, Any]],
    *,
    schedule_date: str | None = None,
    threshold: float = PREDICTOR_COVERAGE_WARN_THRESHOLD,
) -> list[str]:
    warnings: list[str] = []
    for league, summary in sorted(coverage_by_league.items()):
        game_count = summary.get("gameCount") or 0
        if game_count <= 0:
            continue
        predictor_pct = (summary.get("pct") or {}).get("espnPredictor", 0.0)
        if predictor_pct < threshold:
            date_note = f" on {schedule_date}" if schedule_date else ""
            warnings.append(
                f"{league}{date_note}: ESPN predictor coverage {predictor_pct}% "
                f"({summary['counts']['espnPredictor']}/{game_count}) below {threshold}%"
            )
    return warnings


def emit_ci_warnings(warnings: list[str]) -> None:
    for message in warnings:
        print(f"::warning title=Data coverage::{message}")
