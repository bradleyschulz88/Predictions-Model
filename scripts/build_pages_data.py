#!/usr/bin/env python3
"""Build static JSON payloads for GitHub Pages."""

from __future__ import annotations

import json
import os
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import grade_predictions, record_predictions  # noqa: E402
from data_coverage import coverage_warnings, emit_ci_warnings, summarize_coverage  # noqa: E402
from mlb_data import fetch_dashboard_data, strip_betting_lines_for_display  # noqa: E402
from calibration_params import is_publishable_pick  # noqa: E402
from elo import build_and_write as build_elo_ratings  # noqa: E402
from espn_odds import load_side_market_cache, save_side_market_cache  # noqa: E402
from data_providers.odds_api import (  # noqa: E402
    load_cache as load_odds_api_cache,
    quota_status as odds_api_quota,
    save_cache as save_odds_api_cache,
)
from scripts.backtest_model import write_calibration_report  # noqa: E402
from schedule_dates import default_game_date, get_schedule_timezone, schedule_dates_for_league  # noqa: E402
from sports_config import get_league, list_league_ids  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "data"


def dates_for_league(league: str) -> list[str]:
    return schedule_dates_for_league(league)


def include_enrichment_for_date(date_value: str, default_date: str) -> bool:
    """All snapshot dates receive full enrichment in CI builds."""
    return True


def build_league_payload(
    league: str,
    date_value: str,
    *,
    include_enrichment: bool,
    include_odds: bool,
    verify_ssl: bool = True,
) -> dict:
    print(
        f"Building {league} for {date_value} (enrichment={include_enrichment}, odds={include_odds}, ssl={verify_ssl})...",
        flush=True,
    )
    return fetch_dashboard_data(
        league=league,
        date=date_value,
        source="espn",
        include_odds=include_odds,
        include_enrichment=include_enrichment,
        retries=2,
        retry_delay=0.5,
        verify_ssl=verify_ssl,
    )


def build_league_payload_resilient(
    league: str,
    date_value: str,
    *,
    include_enrichment: bool,
    include_odds: bool,
) -> dict:
    """Try verified TLS first; retry without verification on certificate errors."""
    try:
        return build_league_payload(
            league,
            date_value,
            include_enrichment=include_enrichment,
            include_odds=include_odds,
            verify_ssl=True,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "ssl" not in message and "certificate" not in message:
            raise
        print(f"Warning: SSL error for {league} {date_value}, retrying without verify: {exc}", flush=True)
        return build_league_payload(
            league,
            date_value,
            include_enrichment=include_enrichment,
            include_odds=include_odds,
            verify_ssl=False,
        )


def _play_from_game(league_id: str, label: str, game: dict) -> dict:
    """Flatten a game into what the landing page ranks and renders."""
    prediction = game.get("prediction") or {}
    value = prediction.get("value") or {}
    probabilities = prediction.get("probabilities") or {}
    consensus = ((probabilities.get("implied") or {}).get("consensus")) or {}
    side = prediction.get("predictedSide")

    market_pct = None
    if side == "home":
        market_pct = consensus.get("homePct")
    elif side == "away":
        market_pct = consensus.get("awayPct")
    elif side == "draw":
        market_pct = consensus.get("drawPct")

    best_bet = prediction.get("bestBet") or {}
    headline = best_bet.get("pick")

    return {
        "league": league_id,
        "leagueLabel": label,
        "eventId": game.get("eventId"),
        "matchup": game.get("matchup"),
        "homeTeam": game.get("homeTeam"),
        "awayTeam": game.get("awayTeam"),
        "startDate": game.get("startDate"),
        "pick": prediction.get("predictedWinner"),
        "pickSide": side,
        "outcomeLabel": prediction.get("outcomeLabel"),
        "confidence": prediction.get("confidence"),
        "homeWinPct": prediction.get("homeWinPct"),
        "awayWinPct": prediction.get("awayWinPct"),
        "drawWinPct": prediction.get("drawWinPct"),
        "marketPct": market_pct,
        # None when the game is unpriced; there is no EV without a price.
        "value": value or None,
        # Every priced market on this game ranked by EV, and which one is worth
        # backing -- see mlb_predictions.select_best_bet.
        "bestBet": best_bet or None,
        # The bet itself, when it is not the moneyline. The fields above stay
        # moneyline-shaped because the model-vs-market rail is a moneyline
        # comparison; these say what to actually place.
        "betMarket": headline.get("market") if headline else None,
        "betLabel": headline.get("pick") if headline else None,
        # Ranked on the best available bet rather than always the moneyline.
        # Leading the board with a game's moneyline EV while its total was the
        # better bet ranked it by a number that was not the wager on offer.
        "evPct": headline.get("evPct") if headline else value.get("evPct"),
        "kellyPct": headline.get("kellyPct") if headline else value.get("kellyPct"),
        "odds": headline.get("odds") if headline else value.get("odds"),
        "breakEvenPct": headline.get("breakEvenPct") if headline else value.get("breakEvenPct"),
        # Kept separately so the card can still show what the moneyline alone
        # was worth, even when another market won the headline.
        "moneylineEvPct": value.get("evPct"),
    }


def build_overview(payloads: dict[str, dict]) -> dict:
    """Rank every published pick across every league by expected value.

    Confidence is the wrong ranking key: an 89% pick at a price that already
    assumes 89% is not a bet, and the same three points of edge pays +4.0% per
    unit at -300 and +10.5% at +250. EV is what decides.
    """
    league_summaries: list[dict] = []
    plays: list[dict] = []

    for league_id, payload in payloads.items():
        label = payload.get("leagueLabel", league_id)
        games = payload.get("games") or []
        publishable = [
            game for game in games if is_publishable_pick(game.get("prediction"), league_id)
        ]
        league_plays = [_play_from_game(league_id, label, game) for game in publishable]
        plays.extend(league_plays)

        priced = [play for play in league_plays if play.get("evPct") is not None]
        # Lead each league with its best-priced play, or its most confident pick
        # when nothing in that league has odds at all.
        if priced:
            best = max(priced, key=lambda play: play["evPct"])
        elif league_plays:
            best = max(league_plays, key=lambda play: play.get("confidence") or 0)
        else:
            best = None

        league_summaries.append(
            {
                "id": league_id,
                "label": label,
                "scheduleDate": payload.get("scheduleDate"),
                "gameCount": payload.get("gameCount", 0),
                "pickCount": len(league_plays),
                "pricedCount": len(priced),
                "best": best,
                "topPick": payload.get("topPick"),
                # An empty slate and a failed build both render as zero games,
                # which makes an out-of-season league look broken. Carry the
                # reason so the dashboard can tell the two apart.
                "error": payload.get("error"),
                "degraded": payload.get("degraded"),
                # Whether a price source exists for this league at all, so the
                # board can tell "no feed has ever covered this" apart from
                # "the feed failed today". They look identical otherwise.
                "priceCoverage": payload.get("priceCoverage"),
            }
        )

    worth_backing = sorted(
        [play for play in plays if (play.get("evPct") or 0) > 0],
        key=lambda play: play["evPct"],
        reverse=True,
    )
    passed_on = sorted(
        [play for play in plays if play.get("evPct") is not None and play["evPct"] <= 0],
        key=lambda play: play["evPct"],
        reverse=True,
    )
    unpriced = [play for play in plays if play.get("evPct") is None]

    return {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "leagues": league_summaries,
        "worthBacking": worth_backing,
        "passedOn": passed_on[:10],
        "unpriced": sorted(unpriced, key=lambda play: play.get("confidence") or 0, reverse=True)[:10],
        "summary": {
            "picks": len(plays),
            "priced": len(plays) - len(unpriced),
            "positiveEv": len(worth_backing),
            "bestEvPct": worth_backing[0]["evPct"] if worth_backing else None,
            "suggestedUnits": round(sum(play.get("kellyPct") or 0 for play in worth_backing) / 100.0, 2),
            "unpriced": len(unpriced),
        },
        # Kept so anything still reading the old shape does not break.
        "topPicksOverall": sorted(
            plays, key=lambda play: play.get("confidence") or 0, reverse=True
        )[:8],
    }


def report_injury_scorer(payloads: dict[str, dict]) -> None:
    """Say how many teams carried an injury score.

    This used to report whether the NVIDIA importance step ran, because the key
    was optional and a green build was therefore worthless as evidence it
    worked. That scorer is gone -- injuryDiff and injurySeverityDiff both made
    walk-forward log loss worse, so the metered dependency was feeding a feature
    the data kept declining -- and the deterministic score cannot silently fail
    the way a remote call could. What is left is a coverage count, which is
    still worth having: zero teams scored would mean the injury feed broke.
    """
    scored = 0
    for payload in payloads.values():
        for game in payload.get("games") or []:
            enrichment = game.get("enrichment") or {}
            for side in ("home", "away"):
                if (enrichment.get(f"{side}InjurySeverity") or {}).get("source") == "deterministic":
                    scored += 1

    if not scored:
        print("Injury scorer: no teams with injuries to score", flush=True)
        return
    print(f"Injury scorer: {scored} teams scored", flush=True)


def _report_odds_api_quota() -> None:
    """Print the remaining credit balance, live or carried.

    The free plan is 500 credits a month and running out is silent: calls start
    failing, the failure is cached, prices just stop appearing. So the balance
    needs to be in the log of every build, not only the one build in six that
    actually spends a credit.

    Measured across 11 Aug, six sampled builds between 03:01Z and 23:53Z were
    all pure cache hits. None made a call, so none had headers, so the figure
    never appeared -- correct behaviour that added up to the same invisibility
    the reading was added to end. The balance now carries in the disk cache and
    is stamped with when it was taken.
    """
    quota = odds_api_quota()
    if not quota:
        print(
            "Odds API quota: not known yet -- no call has been made since the cache "
            "was last cleared, so the API has not reported a balance",
            flush=True,
        )
        return

    taken = quota.get("asOf")
    if taken:
        hours = (time.time() - taken) / 3600
        age = "read this build" if hours < 0.5 else f"read {hours:.1f}h ago"
    else:
        age = "age unknown"
    figures = {key: value for key, value in quota.items() if key != "asOf"}
    print(f"Odds API quota: {json.dumps(figures)} ({age})", flush=True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Side-market prices carry over from the previous build when the workflow
    # restores the cache file. Without this the hour-long TTL never applies --
    # CI starts a fresh interpreter every thirty minutes, so every game was
    # re-fetched every build, and that request volume is why the pass that
    # prices MLB runlines was switched off in the first place.
    restored = load_side_market_cache()
    if restored:
        print(f"Odds: restored {restored} cached side-market prices", flush=True)

    # Same reason, and here it is money rather than politeness: The Odds API
    # is metered against 500 credits a month, and without this its six-hour
    # cache never survives to be hit, so every build spends afresh.
    restored_api = load_odds_api_cache()
    if restored_api:
        print(f"Odds: restored {restored_api} cached Odds API slates", flush=True)

    manifest: dict = {"builtAt": None, "leagues": []}
    primary_payloads: dict[str, dict] = {}
    payloads_for_accuracy: list[dict] = []
    coverage_reports: dict[str, dict] = {}

    # Elo has to exist BEFORE anything is predicted, because predict_game logs
    # the pre-game rating gap as a feature. elo_ratings.json is a derivative and
    # therefore gitignored, so a fresh CI checkout has no ratings file at all --
    # which meant rating_edge returned None for every game and eloEdge was
    # logged as null on every prediction ever made. The candidate could never
    # accumulate the coverage needed to re-test it.
    #
    # No leakage: this replays only results already graded, and today's games
    # are not among them.
    build_elo_ratings(OUTPUT_DIR)

    for league in list_league_ids():
        league_config = get_league(league)
        default_date = default_game_date(league)
        available_dates = dates_for_league(league)
        date_files: dict[str, str] = {}
        primary_payload: dict | None = None

        for date_value in available_dates:
            include_enrichment = True
            include_odds = True
            try:
                payload = build_league_payload_resilient(
                    league,
                    date_value,
                    include_enrichment=include_enrichment,
                    include_odds=include_odds,
                )
            except Exception as exc:
                print(f"Warning: failed to build {league} {date_value}: {exc}", flush=True)
                payload = {
                    "league": league,
                    "leagueLabel": league_config.label,
                    "scheduleDate": date_value,
                    "games": [],
                    "gameCount": 0,
                    "error": str(exc),
                    "fetchedAt": datetime.now(timezone.utc).isoformat(),
                }

            display_payload = strip_betting_lines_for_display(payload)
            dated_name = f"{league}_{date_value}.json"
            dated_path = OUTPUT_DIR / dated_name
            dated_path.write_text(json.dumps(display_payload, indent=2, default=str), encoding="utf-8")
            date_files[date_value] = f"data/{dated_name}"
            print(f"Wrote {dated_path} ({payload.get('gameCount', 0)} games)", flush=True)
            if payload.get("gameCount", 0) > 0:
                payloads_for_accuracy.append(payload)
                coverage_reports[f"{league}:{date_value}"] = summarize_coverage(payload.get("games") or [])

            if date_value == default_date:
                primary_payload = display_payload
                (OUTPUT_DIR / f"{league}.json").write_text(
                    json.dumps(display_payload, indent=2, default=str),
                    encoding="utf-8",
                )

        if primary_payload is None:
            primary_payload = {
                "league": league,
                "leagueLabel": league_config.label,
                "games": [],
                "gameCount": 0,
                "scheduleDate": default_date,
                "fetchedAt": datetime.now(timezone.utc).isoformat(),
            }

        primary_payloads[league] = primary_payload
        manifest["leagues"].append(
            {
                "id": league,
                "label": league_config.label,
                "espnPath": league_config.espn_path,
                "scheduleTimezone": get_schedule_timezone(league),
                "scheduleDate": primary_payload.get("scheduleDate"),
                "defaultDate": default_date,
                "availableDates": available_dates,
                "dateFiles": date_files,
                "gameCount": primary_payload.get("gameCount", 0),
                "dataCoverage": summarize_coverage(primary_payload.get("games") or []),
                "file": f"data/{league}.json",
                "error": primary_payload.get("error"),
            }
        )

    report_injury_scorer(primary_payloads)

    record_predictions(OUTPUT_DIR, payloads_for_accuracy)
    accuracy = grade_predictions(OUTPUT_DIR)
    # Rebuild again now that today's results have graded, so the published
    # table is current. The pre-prediction build above is the one the feature
    # reads; this one keeps the file itself up to date.
    build_elo_ratings(OUTPUT_DIR)
    write_calibration_report(OUTPUT_DIR)
    overview = build_overview(primary_payloads)

    default_coverage = {
        league: summarize_coverage(primary_payloads[league].get("games") or [])
        for league in primary_payloads
    }
    manifest["dataCoverage"] = default_coverage
    manifest["dataCoverageByDate"] = coverage_reports
    for warning in coverage_warnings(default_coverage):
        print(f"Warning: {warning}", flush=True)
    emit_ci_warnings(coverage_warnings(default_coverage))

    manifest["accuracy"] = accuracy.get("summary")
    manifest["builtAt"] = datetime.now(timezone.utc).isoformat()
    manifest["liveScoreRefreshSeconds"] = 90
    manifest["snapshotNote"] = (
        "Predictions refresh every 30 minutes on GitHub Actions. Live scores auto-refresh every 90s in your browser."
    )

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")

    saved = save_side_market_cache()
    if saved:
        print(f"Odds: saved {saved} side-market prices for the next build", flush=True)

    saved_api = save_odds_api_cache()
    if saved_api:
        print(f"Odds: saved {saved_api} Odds API slates for the next build", flush=True)
    _report_odds_api_quota()

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
