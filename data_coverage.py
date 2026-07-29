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

# Warn when a slate publishes almost nothing. Set low on purpose: a genuinely
# flat slate can legitimately produce few picks, so this fires only when the
# board is effectively empty, which is a threshold problem rather than a weak day.
PUBLISH_RATE_WARN_THRESHOLD = 25.0

# Small slates swing wildly -- 1 of 3 is 33% and means nothing.
MIN_SLATE_FOR_PUBLISH_WARNING = 8


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
    # How many of these games actually reached the board. Kept alongside data
    # coverage because "the model had the data" and "the user saw a pick" are
    # different failures and only one of them was ever being watched.
    from calibration_params import is_publishable_pick

    published = sum(
        1
        for game in games
        if is_publishable_pick(game.get("prediction"), game.get("league"))
    )

    return {
        "gameCount": total,
        "counts": counts,
        "pct": pct,
        "published": published,
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
        date_note = f" on {schedule_date}" if schedule_date else ""

        # Only warn where the feed exists. ESPN publishes a win predictor for the
        # US major leagues and not for Australian football or soccer, so AFL
        # logged a 0% coverage warning on every build -- permanent noise, which
        # is worse than silence because it trains you to skip the annotations.
        predictor_pct = (summary.get("pct") or {}).get("espnPredictor", 0.0)
        if _expects_predictor(league) and predictor_pct < threshold:
            warnings.append(
                f"{league}{date_note}: ESPN predictor coverage {predictor_pct}% "
                f"({summary['counts']['espnPredictor']}/{game_count}) below {threshold}%"
            )

        # A league configured with an odds source that lands zero prices is
        # broken, not unpriced -- and it fails silently, because
        # merge_sbr_odds_into_games swallows SBR errors so a missing board
        # cannot destroy the schedule. Without this check the failure is
        # invisible: WNBA logged 115 picks with no price on any of them while
        # carrying a perfectly good sbr_odds_slug.
        if _expects_odds(league) and not (summary.get("counts") or {}).get("impliedOdds"):
            warnings.append(
                f"{league}{date_note}: has an odds source configured but priced "
                f"0/{game_count} games -- the slug or the team-name match is broken, "
                f"not the market"
            )

        # A publish bar that swallows the slate is a broken bar, not a quiet day.
        #
        # This is the check that was missing when MLB carried a 65 threshold: it
        # was measured against a distribution that then shifted 9.6 points down
        # under it, so it went from excluding the bottom quartile to withholding
        # 90% of games -- 100% on some days -- and nothing said a word. Every
        # build looked green because an empty board is not an error.
        published = summary.get("published")
        if published is not None and game_count >= MIN_SLATE_FOR_PUBLISH_WARNING:
            published_pct = published / game_count * 100
            if published_pct < PUBLISH_RATE_WARN_THRESHOLD:
                warnings.append(
                    f"{league}{date_note}: published only {published}/{game_count} picks "
                    f"({published_pct:.0f}%) -- check the publish bar against the current "
                    f"confidence distribution before assuming the slate is just weak"
                )
    return warnings


def _expects_predictor(league: str) -> bool:
    """True when ESPN publishes a win predictor for this league."""
    try:
        from sports_config import get_league

        return bool(get_league(league).supports_espn_predictor)
    except (ImportError, ValueError):
        # Retired or unknown leagues carry no expectation.
        return False


def _expects_odds(league: str) -> bool:
    """True when the league is configured to have a betting market."""
    try:
        from sports_config import get_league

        return bool(get_league(league).supports_sbr_odds)
    except (ImportError, ValueError):
        # Retired or unknown leagues carry no expectation.
        return False


def emit_ci_warnings(warnings: list[str]) -> None:
    for message in warnings:
        print(f"::warning title=Data coverage::{message}")
