"""Track prediction accuracy and model pick results (inbuilt bet tracker)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data_providers.utils import team_match_score
from espn_client import fetch_scoreboard, parse_scoreboard
from mlb_predictions import _line_odds_value, american_odds_to_implied
from calibration_params import is_publishable_pick
from schedule_dates import league_schedule_date
from sports_config import list_league_ids

ACCURACY_FILE = "accuracy.json"
LOG_FILE = "predictions_log.json"
LOOKBACK_DAYS = 30
DEFAULT_STAKE_UNITS = 1.0

# The log is rewritten and committed every 30 minutes, so it grows without
# bound. Keep enough history to fit on and evaluate against, and drop the rest.
MAX_LOGGED_PREDICTIONS = 5000


def american_odds_profit(odds: int | float, won: bool, stake: float = DEFAULT_STAKE_UNITS) -> float:
    if not won:
        return -stake
    value = float(odds)
    if value < 0:
        return round(stake * (100.0 / abs(value)), 3)
    return round(stake * (value / 100.0), 3)


def extract_pick_american_odds(game: dict[str, Any], predicted_side: str | None) -> int | None:
    if predicted_side not in {"home", "away", "draw"}:
        return None
    key_options = {
        "home": ("home", "homeOdds"),
        "away": ("away", "awayOdds"),
        "draw": ("draw", "drawOdds"),
    }[predicted_side]

    for line in game.get("lines") or []:
        if "MoneyLine" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        odds = _line_odds_value(current, *key_options)
        if odds is not None:
            return int(odds)
    return None


def _compute_streak(results: list[dict[str, Any]]) -> dict[str, Any]:
    current_type: str | None = None
    current_length = 0
    best_win = 0
    best_loss = 0

    for item in results:
        if item.get("status") != "graded":
            continue
        outcome = "win" if item.get("correct") else "loss"
        if current_type == outcome:
            current_length += 1
        else:
            current_type = outcome
            current_length = 1
        if outcome == "win":
            best_win = max(best_win, current_length)
        else:
            best_loss = max(best_loss, current_length)

    return {
        "current": current_length if current_type else 0,
        "type": current_type,
        "bestWin": best_win,
        "bestLoss": best_loss,
    }


def _summary_bucket() -> dict[str, Any]:
    return {
        "correct": 0,
        "total": 0,
        "pct": None,
        "units": 0.0,
        "roiPct": None,
        "pending": 0,
    }


def _accumulate_summary(bucket: dict[str, Any], item: dict[str, Any]) -> None:
    if item.get("status") != "graded":
        bucket["pending"] = bucket.get("pending", 0) + 1
        return
    bucket["total"] += 1
    if item.get("correct"):
        bucket["correct"] += 1
    bucket["units"] = round(bucket.get("units", 0.0) + float(item.get("units") or 0.0), 3)
    if bucket["total"]:
        bucket["pct"] = round(bucket["correct"] / bucket["total"] * 100, 1)
        bucket["roiPct"] = round(bucket["units"] / bucket["total"] * 100, 1)


def _build_pick_record(
    *,
    pending: dict[str, Any],
    game: dict[str, Any] | None = None,
    actual: str | None = None,
    correct: bool | None = None,
    check_date: str | None = None,
    status: str = "pending",
) -> dict[str, Any]:
    record = {
        "eventId": pending.get("eventId"),
        "league": pending.get("league"),
        "scheduleDate": pending.get("scheduleDate"),
        "matchup": pending.get("matchup"),
        "predicted": pending.get("predictedWinner"),
        "predictedSide": pending.get("predictedSide"),
        "outcomeLabel": pending.get("outcomeLabel"),
        "confidence": pending.get("confidence"),
        "rawConfidence": pending.get("rawConfidence"),
        "rawHomeWinPct": pending.get("rawHomeWinPct"),
        "probabilityMethod": pending.get("probabilityMethod"),
        "evPct": pending.get("evPct"),
        "kellyPct": pending.get("kellyPct"),
        "pickOdds": pending.get("pickOdds"),
        "openingOdds": pending.get("openingOdds"),
        "openingSide": pending.get("openingSide"),
        "clvPct": closing_line_value(pending),
        "status": status,
        "actual": actual,
        "correct": correct,
        "units": None,
        "homeScore": None,
        "awayScore": None,
        "gradedAt": None,
        "date": check_date or pending.get("scheduleDate"),
    }
    if status == "graded" and game is not None and correct is not None:
        record["actual"] = actual
        record["correct"] = correct
        record["homeScore"] = game.get("homeScore")
        record["awayScore"] = game.get("awayScore")
        odds = pending.get("pickOdds")
        if odds is not None:
            record["units"] = american_odds_profit(odds, correct)
        record["gradedAt"] = datetime.now(timezone.utc).isoformat()
    return record


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
            if not is_publishable_pick(prediction):
                continue

            current_odds = extract_pick_american_odds(game, prediction.get("predictedSide"))
            existing = (log.get("predictions") or {}).get(event_id) or {}
            # Builds run every 30 minutes, so the first price seen for a game is
            # effectively the opening line and the last one before it starts is
            # the closing line. Keeping both is what makes closing line value
            # measurable, and CLV predicts long-run profitability better than
            # win rate does.
            opening_odds = existing.get("openingOdds")
            if opening_odds is None:
                opening_odds = current_odds
            opening_side = existing.get("openingSide") or prediction.get("predictedSide")

            log["predictions"][event_id] = {
                "eventId": event_id,
                "league": league,
                "scheduleDate": schedule_date,
                "matchup": game.get("matchup"),
                "homeTeam": game.get("homeTeam"),
                "awayTeam": game.get("awayTeam"),
                "predictedWinner": prediction.get("predictedWinner"),
                "predictedSide": prediction.get("predictedSide"),
                "outcomeLabel": prediction.get("outcomeLabel"),
                "confidence": prediction.get("confidence"),
                # Pre-calibration confidence and the path that produced it.
                # Calibration is fitted on these, never on the published
                # confidence, which a previous calibration already adjusted.
                "rawConfidence": prediction.get("rawConfidence"),
                "rawHomeWinPct": prediction.get("rawHomeWinPct"),
                "probabilityMethod": prediction.get("probabilityMethod"),
                # Expected value at pick time, for grading the claim later.
                "evPct": (prediction.get("value") or {}).get("evPct"),
                "kellyPct": (prediction.get("value") or {}).get("kellyPct"),
                # pickOdds is the latest price seen; by grade time it is the
                # closing line. openingOdds is pinned to the first one.
                "pickOdds": current_odds,
                "openingOdds": opening_odds,
                "openingSide": opening_side,
                "features": prediction.get("features"),
                "recordedAt": payload.get("fetchedAt"),
            }

    _prune_log(log)
    _save_json(log_path, log)


def _prune_log(log: dict[str, Any]) -> None:
    """Drop the oldest entries so the committed log stops growing forever."""
    predictions = log.get("predictions") or {}
    if len(predictions) <= MAX_LOGGED_PREDICTIONS:
        return
    ordered = sorted(
        predictions.items(),
        key=lambda item: (item[1].get("scheduleDate") or "", item[0]),
        reverse=True,
    )
    log["predictions"] = dict(ordered[:MAX_LOGGED_PREDICTIONS])


def closing_line_value(record: dict[str, Any]) -> float | None:
    """Percentage points of edge captured against the closing line.

    Positive means the pick was taken at a better price than the market settled
    on. Over a few hundred bets this tracks long-run profitability more reliably
    than win rate, because it measures the decision rather than the outcome.
    """
    opening, closing = record.get("openingOdds"), record.get("pickOdds")
    if opening is None or closing is None:
        return None
    # Opening and closing must describe the same side to be comparable.
    if record.get("openingSide") and record.get("openingSide") != record.get("predictedSide"):
        return None
    try:
        open_implied = american_odds_to_implied(int(opening))
        close_implied = american_odds_to_implied(int(closing))
    except (TypeError, ValueError):
        return None
    # Beating the close means paying a lower implied probability than the market.
    return round((close_implied - open_implied) * 100, 2)


def _prediction_matches_actual(predicted: str | None, actual: str | None) -> bool:
    if not predicted or not actual:
        return False
    if predicted == actual:
        return True
    return team_match_score(predicted, actual) >= 0.92


def _winner_from_game(game: dict[str, Any]) -> str | None:
    if game.get("isVoided") or game.get("isPostponed") or game.get("isCanceled") or game.get("isWashedOut"):
        return None
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
        {
            "updatedAt": None,
            "summary": {
                "last7Days": _summary_bucket(),
                "allTime": _summary_bucket(),
                "byLeague": {},
                "streak": {"current": 0, "type": None, "bestWin": 0, "bestLoss": 0},
            },
            "recentResults": [],
            "pendingPicks": [],
            "picksByEventId": {},
        },
    )

    graded_ids: set[str] = {
        event_id
        for event_id, record in (accuracy.get("picksByEventId") or {}).items()
        if record.get("status") == "graded"
    }
    picks_by_event: dict[str, dict[str, Any]] = dict(accuracy.get("picksByEventId") or {})
    skipped_dates: list[dict[str, str]] = []

    predictions = log.get("predictions") or {}
    dates_to_check: set[tuple[str, str]] = set()
    if predictions:
        for league in list_league_ids():
            for day_offset in range(0, LOOKBACK_DAYS + 1):
                dates_to_check.add((league, league_schedule_date(league, -day_offset)))

        for event_id, pending in predictions.items():
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
        except Exception as exc:
            skipped_dates.append({"league": league, "date": check_date, "error": str(exc)})
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
            record = _build_pick_record(
                pending=pending,
                game=game,
                actual=actual,
                correct=correct,
                check_date=check_date,
                status="graded",
            )
            picks_by_event[event_id] = record
            graded_ids.add(event_id)

    for event_id, pending in predictions.items():
        if event_id in picks_by_event:
            continue
        picks_by_event[event_id] = _build_pick_record(pending=pending, status="pending")

    all_results = sorted(
        picks_by_event.values(),
        key=lambda item: (item.get("gradedAt") or item.get("scheduleDate") or "", item.get("eventId") or ""),
        reverse=True,
    )
    recent_results = [item for item in all_results if item.get("status") == "graded"][:100]
    pending_picks = [item for item in all_results if item.get("status") == "pending"][:50]

    cutoff_dates = {league_schedule_date(league, -LOOKBACK_DAYS) for league in list_league_ids()}
    earliest_cutoff = min(cutoff_dates) if cutoff_dates else (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    window = [
        item
        for item in all_results
        if item.get("status") == "graded" and (item.get("date") or item.get("scheduleDate") or "") >= earliest_cutoff
    ]

    last7 = _summary_bucket()
    all_time = _summary_bucket()
    by_league: dict[str, dict[str, Any]] = {}

    for item in all_results:
        if item.get("status") == "graded":
            _accumulate_summary(all_time, item)
        elif item.get("status") == "pending":
            all_time["pending"] = all_time.get("pending", 0) + 1

    for item in window:
        _accumulate_summary(last7, item)
        league = item.get("league") or "unknown"
        bucket = by_league.setdefault(league, _summary_bucket())
        bucket["total"] += 1
        if item.get("correct"):
            bucket["correct"] += 1
        bucket["units"] = round(bucket.get("units", 0.0) + float(item.get("units") or 0.0), 3)
        if item.get("pickOdds") is not None:
            bucket["priced"] = bucket.get("priced", 0) + 1
        if bucket["total"]:
            bucket["pct"] = round(bucket["correct"] / bucket["total"] * 100, 1)
            bucket["roiPct"] = round(bucket["units"] / bucket["total"] * 100, 1)

    # A win rate with no price behind it is not comparable to one with a price.
    # AFL has no odds source, so its ROI reads 0.0% and
    # looks like break-even rather than "not measurable".
    for bucket in by_league.values():
        priced = bucket.get("priced", 0)
        bucket["priced"] = priced
        bucket["unpriced"] = bucket["total"] - priced
        bucket["pricedPct"] = round(priced / bucket["total"] * 100, 1) if bucket["total"] else None
        if not priced:
            bucket["roiPct"] = None
            bucket["roiNote"] = "No odds available for this league; ROI is not measurable."

    last7["pending"] = sum(
        1 for item in pending_picks if (item.get("scheduleDate") or "") >= earliest_cutoff
    )

    clv_values = [
        float(item["clvPct"]) for item in window if item.get("clvPct") is not None
    ]
    clv_summary = {
        "picks": len(clv_values),
        "avgPct": round(sum(clv_values) / len(clv_values), 2) if clv_values else None,
        "beatCloseP": (
            round(sum(1 for value in clv_values if value > 0) / len(clv_values) * 100, 1)
            if clv_values
            else None
        ),
        "note": "Closing line value predicts long-run profitability better than win rate.",
    }

    streak = _compute_streak(recent_results)

    accuracy = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "last7Days": last7,
            "allTime": all_time,
            "byLeague": by_league,
            "streak": streak,
            "closingLineValue": clv_summary,
        },
        "recentResults": recent_results,
        "pendingPicks": pending_picks,
        "picksByEventId": picks_by_event,
        "skippedDates": skipped_dates,
    }
    _save_json(accuracy_path, accuracy)
    return accuracy
