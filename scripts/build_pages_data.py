#!/usr/bin/env python3
"""Build static JSON payloads for GitHub Pages."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import grade_predictions, record_predictions  # noqa: E402
from mlb_data import fetch_dashboard_data, strip_betting_lines_for_display  # noqa: E402
from scripts.backtest_model import write_calibration_report  # noqa: E402
from schedule_dates import default_game_date, get_schedule_timezone, schedule_dates_for_league  # noqa: E402
from sports_config import LEAGUES, get_league, list_league_ids  # noqa: E402

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
) -> dict:
    print(
        f"Building {league} for {date_value} (enrichment={include_enrichment}, odds={include_odds})...",
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
        verify_ssl=True,
    )


def build_overview(payloads: dict[str, dict]) -> dict:
    league_summaries: list[dict] = []
    top_picks: list[dict] = []

    for league_id, payload in payloads.items():
        games = payload.get("games") or []
        top_game = games[0] if games else None
        league_summaries.append(
            {
                "id": league_id,
                "label": payload.get("leagueLabel", league_id),
                "scheduleDate": payload.get("scheduleDate"),
                "gameCount": payload.get("gameCount", 0),
                "topPick": payload.get("topPick"),
                "topConfidence": (top_game or {}).get("prediction", {}).get("confidence"),
            }
        )
        for game in games[:3]:
            prediction = game.get("prediction") or {}
            top_picks.append(
                {
                    "league": league_id,
                    "leagueLabel": payload.get("leagueLabel", league_id),
                    "matchup": game.get("matchup"),
                    "pick": prediction.get("outcomeLabel"),
                    "confidence": prediction.get("confidence"),
                    "confidenceLabel": prediction.get("confidenceLabel"),
                    "eventId": game.get("eventId"),
                }
            )

    top_picks.sort(key=lambda item: item.get("confidence") or 0, reverse=True)
    return {
        "builtAt": datetime.now(timezone.utc).isoformat(),
        "leagues": league_summaries,
        "topPicksOverall": top_picks[:8],
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"builtAt": None, "leagues": []}
    primary_payloads: dict[str, dict] = {}
    payloads_for_accuracy: list[dict] = []

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
                payload = build_league_payload(
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
                "file": f"data/{league}.json",
                "error": primary_payload.get("error"),
            }
        )

    record_predictions(OUTPUT_DIR, payloads_for_accuracy)
    accuracy = grade_predictions(OUTPUT_DIR)
    write_calibration_report(OUTPUT_DIR)
    overview = build_overview(primary_payloads)

    manifest["accuracy"] = accuracy.get("summary")
    manifest["builtAt"] = datetime.now(timezone.utc).isoformat()
    manifest["liveScoreRefreshSeconds"] = 90
    manifest["snapshotNote"] = (
        "Predictions refresh every 30 minutes on GitHub Actions. Live scores auto-refresh every 90s in your browser."
    )

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
