#!/usr/bin/env python3
"""Build static JSON payloads for GitHub Pages."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from accuracy_tracker import grade_predictions, record_predictions  # noqa: E402
from mlb_data import default_game_date, fetch_dashboard_data  # noqa: E402
from sports_config import list_league_ids  # noqa: E402

OUTPUT_DIR = ROOT / "docs" / "data"


def build_league_payload(league: str) -> dict:
    date_value = default_game_date(league)
    print(f"Building {league} for {date_value}...", flush=True)
    return fetch_dashboard_data(
        league=league,
        date=date_value,
        source="espn",
        include_odds=league == "mlb",
        include_enrichment=True,
        retries=2,
        retry_delay=0.5,
        verify_ssl=True,
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"builtAt": None, "leagues": []}
    payloads: dict[str, dict] = {}

    for league in list_league_ids():
        try:
            payload = build_league_payload(league)
        except Exception as exc:
            print(f"Warning: failed to build {league}: {exc}", flush=True)
            payload = {
                "league": league,
                "games": [],
                "gameCount": 0,
                "error": str(exc),
                "fetchedAt": datetime.now(timezone.utc).isoformat(),
            }

        payloads[league] = payload
        output_path = OUTPUT_DIR / f"{league}.json"
        output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        manifest["leagues"].append(
            {
                "id": league,
                "label": payload.get("leagueLabel", league),
                "scheduleDate": payload.get("scheduleDate"),
                "gameCount": payload.get("gameCount", 0),
                "file": f"data/{league}.json",
                "error": payload.get("error"),
            }
        )
        print(f"Wrote {output_path} ({payload.get('gameCount', 0)} games)", flush=True)

    record_predictions(OUTPUT_DIR, payloads)
    accuracy = grade_predictions(OUTPUT_DIR)
    manifest["accuracy"] = accuracy.get("summary")
    manifest["builtAt"] = datetime.now(timezone.utc).isoformat()
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
