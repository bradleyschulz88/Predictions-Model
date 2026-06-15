"""Track prediction accuracy over time."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data_providers.utils import team_match_score
from espn_client import fetch_scoreboard, parse_scoreboard
from schedule_dates import league_schedule_date
from sports_config import list_league_ids

ACCURACY_FILE = "accuracy.json"
LOG_FILE = "predictions_log.json"
LOOKBACK_DAYS = 7


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def record_predictions(data_dir: Path, payloads: dict[str, dict[str, Any]] | list[dict[str, Any]]) -> None:
    log_path = data_dir / LOG_FILE
    log = _load_json(log_path, {"predictions": {}})

    if isinstance(payloads, dict):
        payload_list = list(payloads.values())
    else:
        payload_list = payloads

    for payload in payload_list:
        league = payload.get("league") or "unknown"
        schedule_date = payload.get("scheduleDate")
        for game in payload.get("games") or []:
            prediction = game.get("prediction") or {}
            event_id = str(game.get("eventId") or "")
            if not event_id or not prediction.get("predictedWinner"):
                continue
            log["predictions"][event_id] = {
                "eventId": event_id,
                "league": league,
                "scheduleDate": schedule_date,
                "matchup": game.get("matchup"),
                "homeTeam": game.get("homeTeam"),
                "awayTeam": game.get("awayTeam"),
                "predictedWinner": prediction.get("predictedWinner"),
                "predictedSide": prediction.get("predictedSide"),
                "confidence": prediction.get("confidence"),
                "recordedAt": payload.get("fetchedAt"),
            }

    _save_json(log_path, log)


def _prediction_matches_actual(predicted: str | None, actual: str | None) -> bool:
    if not predicted or not actual:
        return False
    if predicted == actual:
        return True
    return team_match_score(predicted, actual) >= 0.92


def _winner_from_game(game: dict[str, Any]) -> str | None:
    if not game.get("isFinal"):
        return None
    home_score = game.get("homeScore")
    away_score = game.get("awayScore")
    if home_score is None or away_score is None:
        return None
    try:
        home = int(home_score)
        away = int(away_score)
    except (TypeError, ValueError):
        return None
    if home == away:
        return "Draw"
    return game.get("homeTeam") if home > away else game.get("awayTeam")


def grade_predictions(data_dir: Path, *, verify_ssl: bool = True) -> dict[str, Any]:
    log_path = data_dir / LOG_FILE
    accuracy_path = data_dir / ACCURACY_FILE
    log = _load_json(log_path, {"predictions": {}})
    accuracy = _load_json(
        accuracy_path,
        {"updatedAt": None, "summary": {"last7Days": {"correct": 0, "total": 0, "pct": None}, "byLeague": {}}, "recentResults": []},
    )

    graded_ids: set[str] = {item.get("eventId") for item in accuracy.get("recentResults") or [] if item.get("eventId")}
    new_results: list[dict[str, Any]] = list(accuracy.get("recentResults") or [])

    dates_to_check: set[tuple[str, str]] = set()
    for league in list_league_ids():
        for day_offset in range(0, LOOKBACK_DAYS + 1):
            dates_to_check.add((league, league_schedule_date(league, -day_offset)))

    for event_id, pending in (log.get("predictions") or {}).items():
        if event_id in graded_ids:
            continue
        league = pending.get("league")
        schedule_date = pending.get("scheduleDate")
        if league and schedule_date:
            dates_to_check.add((league, schedule_date))

    for league, check_date in sorted(dates_to_check):
        try:
            scoreboard = fetch_scoreboard(league, check_date, retries=2, retry_delay=0.5, verify_ssl=verify_ssl)
            games = parse_scoreboard(scoreboard, league=league)
        except Exception:
            continue

        for game in games:
            event_id = str(game.get("eventId") or "")
            if not event_id or event_id in graded_ids:
                continue
            pending = log.get("predictions", {}).get(event_id)
            if not pending:
                continue
            actual = _winner_from_game(game)
            if not actual:
                continue

            predicted = pending.get("predictedWinner")
            correct = _prediction_matches_actual(predicted, actual)
            result = {
                "eventId": event_id,
                "date": check_date,
                "league": league,
                "matchup": game.get("matchup") or pending.get("matchup"),
                "predicted": predicted,
                "actual": actual,
                "correct": correct,
                "confidence": pending.get("confidence"),
            }
            new_results.insert(0, result)
            graded_ids.add(event_id)

    new_results = new_results[:100]
    cutoff_dates = {league_schedule_date(league, -LOOKBACK_DAYS) for league in list_league_ids()}
    earliest_cutoff = min(cutoff_dates) if cutoff_dates else (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    window = [item for item in new_results if (item.get("date") or "") >= earliest_cutoff]
    correct = sum(1 for item in window if item.get("correct"))
    total = len(window)

    by_league: dict[str, dict[str, Any]] = {}
    for item in window:
        league = item.get("league") or "unknown"
        bucket = by_league.setdefault(league, {"correct": 0, "total": 0, "pct": None})
        bucket["total"] += 1
        if item.get("correct"):
            bucket["correct"] += 1
    for bucket in by_league.values():
        if bucket["total"]:
            bucket["pct"] = round(bucket["correct"] / bucket["total"] * 100, 1)

    from datetime import datetime, timezone

    accuracy = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "last7Days": {
                "correct": correct,
                "total": total,
                "pct": round(correct / total * 100, 1) if total else None,
            },
            "byLeague": by_league,
        },
        "recentResults": new_results,
    }
    _save_json(accuracy_path, accuracy)
    return accuracy
