"""Data coverage metrics for model inputs across built game payloads."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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

# A league with an odds source that suddenly prices far fewer games has a
# broken source, and until now that only surfaced as a flat board -- the
# existing check below fires at exactly zero, so a source degrading from 15/15
# to 3/15 was invisible.
#
# Measured on the 2026-08-05 build, healthy pricing is effectively total on
# games a book has had time to post: MLB 15/15, 15/15, 15/15, 14/15 across
# 08-01..08-05. So a real degradation is far below this, and 60% will not fire
# on one book dropping a single game.
PRICED_SHARE_WARN_THRESHOLD = 60.0

# Books post lines about a day out. The same build shows MLB 0/11 priced for
# 08-06 and 0/15 for 08-07 while every earlier date is complete -- that is the
# market not having opened yet, not a fault, and counting it would fire this
# warning on every slate the board carries. So only games a price should
# already exist for are counted. Generous on purpose: 36 hours is past the
# observed horizon, so anything still unpriced inside it is a genuine miss.
PRICE_EXPECTED_WITHIN_HOURS = 36

# Same reason as MIN_SLATE_FOR_PUBLISH_WARNING: a share computed on three
# games is noise.
MIN_SLATE_FOR_PRICING_WARNING = 5


def _price_should_exist(game: dict[str, Any], *, now: datetime | None = None) -> bool:
    """True when a book has had time to post a line for this game.

    A game already under way or finished certainly had one. A game far enough
    in the future may legitimately have none, and treating that as a fault is
    how the predictor-coverage check spent weeks crying wolf.
    """
    if game.get("isVoided"):
        return False
    if game.get("isFinal") or game.get("isLive"):
        return True

    reference = now or datetime.now(timezone.utc)
    start = game.get("startDate") or game.get("scheduleDate")
    if not start:
        # No time to reason about. Counting it would let a feed that stops
        # emitting start times quietly switch this check off.
        return True
    try:
        parsed = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed <= reference + timedelta(hours=PRICE_EXPECTED_WITHIN_HOURS)


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


def summarize_coverage(
    games: list[dict[str, Any]], *, now: datetime | None = None
) -> dict[str, Any]:
    total = len(games)
    counts = {flag: 0 for flag in COVERAGE_FLAGS}
    # ESPN publishes the Matchup Predictor before a game and drops it the moment
    # the game is final, so a finished slate cannot carry one and counting it
    # against the whole slate measures the clock, not the feed. Verified live:
    # the 2026-08-05 slate returned predictor coverage 100%, the 2026-08-04
    # slate 0/15 with every game 'Final'.
    predictor_eligible = 0
    predictor_present = 0
    priced_eligible = 0
    priced_present = 0
    for game in games:
        flags = coverage_from_game(game)
        for flag, present in flags.items():
            if present:
                counts[flag] += 1
        if _can_carry_predictor(game):
            predictor_eligible += 1
            if flags.get("espnPredictor"):
                predictor_present += 1
        if _price_should_exist(game, now=now):
            priced_eligible += 1
            if flags.get("impliedOdds"):
                priced_present += 1

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
        # Reported separately from counts/pct so the dashboard's coverage
        # display keeps meaning "of this slate" while the warning can ask the
        # answerable question: of the games that could have a predictor, how
        # many did.
        "predictorEligible": predictor_eligible,
        "predictorPresent": predictor_present,
        "predictorPct": (
            round(predictor_present / predictor_eligible * 100, 1)
            if predictor_eligible
            else None
        ),
        # Same treatment for prices: measured over games a book has had time
        # to post, so a slate that is simply too far out never reads as a
        # broken feed.
        "pricedEligible": priced_eligible,
        "pricedPresent": priced_present,
        "pricedSharePct": (
            round(priced_present / priced_eligible * 100, 1) if priced_eligible else None
        ),
    }


def summarize_league_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_league: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        league = payload.get("league") or "unknown"
        games = payload.get("games") or []
        by_league[league] = summarize_coverage(games)
    return by_league


def _can_carry_predictor(game: dict[str, Any]) -> bool:
    """True while ESPN would still publish a Matchup Predictor for this game.

    The field is pregame-only: it is present on a scheduled game and gone once
    the game is final. Voided games (postponed, cancelled) never get one either.
    """
    if game.get("isFinal") or game.get("isVoided"):
        return False
    status = (game.get("statusType") or "").upper()
    return status not in {"STATUS_FINAL", "STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"}


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
        # Measured over games that could carry a predictor at all. A finished
        # slate has none by construction, and warning about that is the same
        # permanent noise as the AFL case above -- it fired on every overnight
        # build and meant nothing.
        eligible = summary.get("predictorEligible") or 0
        predictor_pct = summary.get("predictorPct")
        if _expects_predictor(league) and eligible and predictor_pct < threshold:
            warnings.append(
                f"{league}{date_note}: ESPN predictor coverage {predictor_pct}% "
                f"({summary.get('predictorPresent', 0)}/{eligible} games still to be "
                f"played) below {threshold}%"
            )

        # A league configured with an odds source that lands zero prices is
        # broken, not unpriced -- and it fails silently, because
        # merge_sbr_odds_into_games swallows SBR errors so a missing board
        # cannot destroy the schedule. Without this check the failure is
        # invisible: WNBA logged 115 picks with no price on any of them while
        # carrying a perfectly good sbr_odds_slug.
        # Gated on eligibility for the same reason as the share check below: a
        # slate three days out is legitimately unpriced everywhere, and this
        # would call that a broken slug on every future date it was handed.
        # Today it only ever sees the default date, so the fault never showed
        # -- which is exactly how it would survive to bite later.
        priced_eligible = summary.get("pricedEligible")
        if priced_eligible is None:
            priced_eligible = game_count
        if (
            _expects_odds(league)
            and priced_eligible
            and not (summary.get("counts") or {}).get("impliedOdds")
        ):
            warnings.append(
                f"{league}{date_note}: has an odds source configured but priced "
                f"0/{priced_eligible} games a book should have posted by now -- the "
                f"slug or the team-name match is broken, not the market"
            )

        # Partial degradation, which the zero-check above cannot see. A source
        # that goes from pricing every game to pricing a third of them is
        # broken in exactly the way that shows up as a thin board days later
        # -- the point is to hear it from the build instead.
        priced_eligible = summary.get("pricedEligible") or 0
        priced_share = summary.get("pricedSharePct")
        if (
            _expects_odds(league)
            and priced_eligible >= MIN_SLATE_FOR_PRICING_WARNING
            and priced_share is not None
            and 0 < priced_share < PRICED_SHARE_WARN_THRESHOLD
        ):
            warnings.append(
                f"{league}{date_note}: priced only {summary.get('pricedPresent', 0)}"
                f"/{priced_eligible} games a book should have posted by now "
                f"({priced_share}%, below {PRICED_SHARE_WARN_THRESHOLD}%) -- a price "
                f"source has degraded, the slate has not"
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
