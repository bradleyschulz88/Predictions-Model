"""Track prediction accuracy and model pick results (inbuilt bet tracker)."""

from __future__ import annotations

import json
import math
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data_providers.utils import team_match_score
from espn_client import fetch_scoreboard, parse_scoreboard
from mlb_predictions import _best_price_for_side, american_odds_to_implied, quote_spread
from calibration_params import is_publishable_pick
from schedule_dates import league_schedule_date
from sports_config import list_league_ids

ACCURACY_FILE = "accuracy.json"
LOG_FILE = "predictions_log.json"
LOOKBACK_DAYS = 30
DEFAULT_STAKE_UNITS = 1.0

# Break-even win rate at the standard -110, used only until a market has real
# prices of its own to measure against.
DEFAULT_BREAK_EVEN_PCT = 52.4


def _implied_break_even(odds: Any) -> float:
    """The win rate a price needs just to return the stake, as a percentage."""
    value = float(odds)
    decimal = 1.0 + (100.0 / abs(value) if value < 0 else value / 100.0)
    return 100.0 / decimal

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
    """The price this pick is graded at -- the same one its EV was computed from.

    This used to walk the lines itself and return the FIRST moneyline it found,
    while the published EV, Kelly stake and board ranking all came from
    _best_price_for_side, which returns the BEST across every book quoting the
    game. The two disagreed on 61% of the graded record, always in the same
    direction: the board advertised an edge at a price the record never booked.
    A 60% pick shown at -136 as +4.1% EV was graded at -155, where it is -1.3%.

    Delegating rather than reimplementing is the point. Two functions picking
    "the" price from the same list is what caused the drift, so there is now
    one, and it carries the outlier guard for both.
    """
    if predicted_side not in {"home", "away", "draw"}:
        return None
    return _best_price_for_side(game.get("lines") or [], predicted_side)


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
        # Games called off. Never counted as a win or a loss, but counted, so a
        # pick that disappears from the record has a stated reason.
        "voided": 0,
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
        # Set once the game is under way, meaning pickOdds is a real pre-game
        # close. Absent means it is only the latest quote seen, which is not
        # the same thing and must not be averaged in as if it were.
        "pickOddsFrozenAt": pending.get("pickOddsFrozenAt"),
        # Rows logged before publishing and logging were separated have no flag
        # and were publishable by definition, so absent means True.
        "published": pending.get("published", True),
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

        # Separate markets on the same game, scored separately. They must never
        # be folded into `correct`, which is the moneyline record, or a good
        # totals week would flatter a bad picking week.
        record["totalResult"] = grade_total(
            pending.get("total"), game.get("homeScore"), game.get("awayScore")
        )
        record["spreadResult"] = grade_spread(
            pending.get("spread"), game.get("homeScore"), game.get("awayScore")
        )
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
            # Log every pick the model makes, publish only the ones that clear
            # the bar. These are different questions and were previously one.
            #
            # MLB's 55-65% band is withheld from the board because it loses
            # money, but it is precisely the band where the fit is most wrong,
            # so censoring it from the training log would stop the model ever
            # learning to correct itself there -- the threshold would entrench
            # the very error it exists to hide. `published` records which side
            # of the bar a pick fell on, so accuracy and ROI can still report on
            # the board alone.
            published = is_publishable_pick(prediction, league)

            current_odds = extract_pick_american_odds(game, prediction.get("predictedSide"))
            existing = (log.get("predictions") or {}).get(event_id) or {}
            # Builds run every 30 minutes, so the first price seen for a game is
            # effectively the opening line and the last one before it starts is
            # the closing line. Keeping both is what makes closing line value
            # measurable, and CLV predicts long-run profitability better than
            # win rate does.
            opening_odds = existing.get("openingOdds")
            opening_at = existing.get("openingOddsAt")
            if opening_odds is None:
                opening_odds = current_odds
                # When the first price was taken. Without it there is no way to
                # ask how far ahead of the game the model committed, which is
                # the leading explanation for its negative closing line value:
                # a side taken days early and drifting out is what an adverse
                # CLV looks like, and that is a timing problem rather than a
                # modelling one. Only meaningful next to scheduleDate.
                if current_odds is not None:
                    opening_at = payload.get("fetchedAt")
            opening_side = existing.get("openingSide") or prediction.get("predictedSide")

            # Two ways the closing price was being destroyed, both silent.
            #
            # Nothing stopped a build from overwriting pickOdds after the game
            # had started, so a build landing mid-game replaced the closing line
            # with an in-play number, and one landing after the final replaced
            # it with whatever the book showed then. Either way CLV was
            # measuring the wrong pair of prices.
            #
            # And any build where the odds fetch came back empty -- a provider
            # blip, a book pulling a market -- wrote None straight over a price
            # already recorded, losing it for good.
            #
            # So: a recorded price is never replaced by nothing, and never
            # updated once the game is under way. The last price seen before
            # the first pitch is the close, which is what CLV needs.
            previous_odds = existing.get("pickOdds")
            started = bool(game.get("isLive") or game.get("isFinal") or game.get("isVoided"))
            frozen_at = existing.get("pickOddsFrozenAt")
            if started and previous_odds is not None:
                pick_odds = previous_odds
                frozen_at = frozen_at or payload.get("fetchedAt")
            elif current_odds is None:
                pick_odds = previous_odds
            else:
                pick_odds = current_odds

            # Freeze the feature vector when the game starts, exactly as the
            # closing price is frozen, and for the same reason.
            #
            # This used to be a plain overwrite on every build. The build
            # re-enriches dates that have already been played, so a graded
            # row's features were whatever the sources said *after* the result
            # -- and anything computed from a source that updates on a result
            # therefore encoded it. h2hDiff was the severe case, reading a
            # season-series score that a single game moves by a third: it
            # reached a standalone AUC of 0.855 against the closing line's
            # 0.640. strengthDiff, which ships, sits at 0.682 on the same test.
            #
            # The consequence is not confined to the reports. The fit trains on
            # these rows and then predicts on clean pre-game ones, so a weight
            # learned from an inflated feature is too large for the value that
            # feature actually carries before a game. Leaked evaluation and
            # degraded predictions, from one missing line.
            #
            # Frozen at the last pre-game observation rather than the first,
            # which is what pickOdds does: the first build to see a game days
            # out may have no odds and thin enrichment, and there is no reason
            # to prefer that to the fullest picture available at first pitch.
            previous_features = existing.get("features")
            features_frozen_at = existing.get("featuresFrozenAt")
            if started and previous_features:
                pinned_features = previous_features
                features_frozen_at = features_frozen_at or payload.get("fetchedAt")
            else:
                pinned_features = prediction.get("features") or previous_features

            # The path between open and close, not just its two endpoints.
            #
            # Line movement is among the most predictive publicly available
            # signals, and this project discarded it on every build: each run
            # fetched a price, compared it to nothing, and overwrote. The feed
            # is already paid for, so keeping the path costs storage and
            # nothing else. Unlike a feature that can be derived later from
            # data on disk, history not recorded now cannot be recovered --
            # which is why this went in before the analysis that will use it.
            #
            # Placed after `started` deliberately: an in-play quote is not part
            # of the pre-game path and would corrupt exactly the measurement
            # this exists to support.
            price_history = _extend_price_history(
                existing.get("priceHistory"),
                current_odds,
                payload.get("fetchedAt"),
                started=started,
            )

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
                "pickOdds": pick_odds,
                # What shopping was worth on this game: how many books quoted
                # it, and the gap in implied probability between the best and
                # the median. The build already takes the best price; until now
                # it never recorded what the alternatives were, so the value of
                # doing so could not be measured. Pinned like openingOdds --
                # written once, because after the game the books are gone.
                "priceSpread": existing.get("priceSpread") or quote_spread(
                    game.get("lines") or [], prediction.get("predictedSide")
                ),
                "openingOdds": opening_odds,
                "openingOddsAt": opening_at,
                "openingSide": opening_side,
                # Every distinct pre-game price seen, oldest first.
                "priceHistory": price_history,
                # When the price stopped being updated. Present means this is a
                # genuine pre-game close rather than the latest quote seen.
                "pickOddsFrozenAt": frozen_at,
                # False means the model made this pick but the board withheld
                # it. Kept for training; excluded from the published record.
                "published": published,
                "features": pinned_features,
                # Present means these features are the last pre-game state and
                # can be trained on. Absent means they were recomputed after
                # the result was known and cannot -- see the comment above.
                "featuresFrozenAt": features_frozen_at,
                # Separate markets on the same game. Logged so they can be
                # graded, which none of them ever has been -- the totals
                # heuristic has been shown on the board since the start without
                # a single scored result behind it.
                "total": _side_market(prediction.get("total"), ("line", "pickSide", "odds")),
                "spread": _side_market(
                    prediction.get("spread"), ("line", "pickSide", "market", "odds")
                ),
                "recordedAt": payload.get("fetchedAt"),
            }

    _prune_log(log)
    _save_json(log_path, log)


def _market_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """Win/loss/push record for one side market, with ROI once any pick in it
    carries a logged price.

    Hit rate excludes pushes, because a returned stake is not a result. Counting
    pushes as half a win -- the other common convention -- would quietly lift a
    break-even record above break-even.

    Units follow the same convention the per-league moneyline summary already
    uses: an unpriced graded pick contributes zero to the total rather than
    being excluded from the denominator, so ROI still reads as "return per
    graded pick" while most of them carry no price yet -- these markets have a
    price only when ESPN's core odds supplied one; SBR has nowhere to carry it
    at all. If literally none of them do, roiPct stays None with a note
    instead of reporting a misleading 0.0%.
    """
    graded = [item[key] for item in results if isinstance(item.get(key), dict)]
    wins = sum(1 for row in graded if row.get("outcome") == "win")
    losses = sum(1 for row in graded if row.get("outcome") == "loss")
    pushes = sum(1 for row in graded if row.get("outcome") == "push")
    decided = wins + losses
    priced_rows = [row for row in graded if row.get("odds") is not None]
    priced = len(priced_rows)
    units = round(sum(float(row.get("units") or 0.0) for row in graded), 3)
    priced_units = round(sum(float(row.get("units") or 0.0) for row in priced_rows), 3)

    # The bar these picks actually had to clear, from the prices they were
    # really taken at rather than an assumed -110. It matters as soon as MLB
    # runlines start carrying prices: those sit nearer +150/-200, where 52.4%
    # is badly wrong. Falls back to -110 only when nothing is priced yet.
    break_even = round(
        sum(_implied_break_even(row["odds"]) for row in priced_rows) / priced, 1
    ) if priced else DEFAULT_BREAK_EVEN_PCT

    def _rate(rows: list[dict[str, Any]]) -> tuple[int, float | None, float | None]:
        """Hit rate and its binomial standard error over one set of rows.

        At these sample sizes the error bar is the difference between "this
        market wins" and "this market might win": 56 decided picks carries
        about 6.3 points, so a 61% record and a 52% record are not
        distinguishable from one another.
        """
        won = sum(1 for row in rows if row.get("outcome") == "win")
        lost = sum(1 for row in rows if row.get("outcome") == "loss")
        settled = won + lost
        if not settled:
            return 0, None, None
        share = won / settled
        return settled, round(share * 100, 1), round(math.sqrt(share * (1 - share) / settled) * 100, 1)

    _, pct, std_err = _rate(graded)

    # The same rate over only the picks that carry a price. This is the one
    # that belongs next to ROI and break-even, and keeping them apart matters:
    # measured 2026-08-05, totals ran 47.8% priced against 90.0% on the ten
    # unpriced picks, so the blended 53.2% sat above a 52.4% break-even while
    # the money went the other way at -7.2%. The hit rate and the return were
    # describing different populations, which reads as a market that wins and
    # loses at the same time. Break-even is derived from prices, so it can only
    # honestly be compared against the picks that had one.
    priced_decided, priced_pct, priced_std_err = _rate(priced_rows)

    # Does the record clear break-even by more than noise? Lower 95% bound,
    # priced picks only, against the bar those picks actually faced. This is
    # the honest answer to "has this market shown it can pick at these prices".
    beats_break_even = bool(
        priced_pct is not None
        and priced_std_err is not None
        and (priced_pct - 1.96 * priced_std_err) > break_even
    )

    summary: dict[str, Any] = {
        "graded": len(graded),
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pct": pct,
        "stdErrPct": std_err,
        "pricedDecided": priced_decided,
        "pricedPct": priced_pct,
        "pricedStdErrPct": priced_std_err,
        "breakEvenPct": break_even,
        "beatsBreakEven": beats_break_even,
        "priced": priced,
        "unpriced": len(graded) - priced,
    }
    if priced:
        summary["units"] = units
        summary["roiPct"] = round(units / len(graded) * 100, 1) if graded else None
        # The same units over only the picks that could have produced them.
        # roiPct divides by every graded pick, so a market with 9 prices out of
        # 56 reports a modest return that is really nine picks' worth of
        # evidence -- understating the priced result and hiding how thin it is.
        summary["pricedUnits"] = priced_units
        summary["pricedRoiPct"] = round(priced_units / priced * 100, 1)
        if priced < len(graded):
            note = (
                f"ROI is over all {len(graded)} graded picks; {len(graded) - priced} of them "
                f"carry no logged price and count as zero return. Over the {priced} priced "
                f"picks alone the return is {priced_units / priced * 100:+.1f}%."
            )
            # Say it outright when the two populations disagree, rather than
            # leaving a reader to notice that a hit rate over every graded pick
            # is sitting beside a return over only the priced ones.
            if priced_pct is not None and pct is not None and abs(priced_pct - pct) >= 3.0:
                note += (
                    f" The {pct}% hit rate covers all {decided} decided picks; over the"
                    f" priced ones alone it is {priced_pct}%, and that is the figure the"
                    f" {break_even}% break-even applies to."
                )
        else:
            note = "Hit rate and ROI both cover the full graded record."
        summary["note"] = note
    else:
        summary["units"] = 0.0
        summary["roiPct"] = None
        summary["pricedUnits"] = 0.0
        summary["pricedRoiPct"] = None
        summary["note"] = "Hit rate only; these markets carry no logged price, so ROI is not measurable."
    return summary


def _side_market(block: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """Just enough of a totals or spread pick to grade it later.

    Deliberately narrow. The full block carries reasoning text that would bloat
    a log committed every 30 minutes, and grading needs only the line and the
    side taken.
    """
    if not isinstance(block, dict):
        return None
    kept = {key: block.get(key) for key in keys if block.get(key) is not None}
    if "line" not in kept or "pickSide" not in kept:
        return None
    return kept


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


# Distinct pre-game prices kept per event. A line that moves more than this in
# one game is not a line, it is a data fault, and the endpoints are preserved
# regardless because openingOdds and pickOdds are stored separately.
MAX_PRICE_OBSERVATIONS = 40


def _extend_price_history(
    existing: Any, odds: Any, at: str | None, *, started: bool
) -> list[dict[str, Any]]:
    """Append a price observation, but only when it is new and pre-game.

    Deduplicated on change rather than sampled per build, which is what keeps
    this affordable. Builds land roughly hourly and a baseball line is
    unchanged across most of them, so storing every observation would grow
    predictions_log.json by thousands of identical rows a week to record
    nothing. Storing only the moves keeps the same information.

    Never appends once the game is under way: an in-play quote is not part of
    the pre-game path, and letting one in would corrupt the exact measurement
    this series exists to support.
    """
    history = [
        entry for entry in (existing or [])
        if isinstance(entry, dict) and entry.get("odds") is not None
    ]
    if started or odds is None:
        return history[-MAX_PRICE_OBSERVATIONS:]
    if history and history[-1].get("odds") == odds:
        return history[-MAX_PRICE_OBSERVATIONS:]
    history.append({"at": at, "odds": odds})
    return history[-MAX_PRICE_OBSERVATIONS:]


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
        # OverflowError is the one that mattered: int(float("inf")) raises it,
        # not ValueError, so a single non-finite price aborted the whole grading
        # run rather than costing one row its CLV. This function is called for
        # every record in the loop.
        open_implied = american_odds_to_implied(int(opening))
        close_implied = american_odds_to_implied(int(closing))
    except (TypeError, ValueError, OverflowError):
        return None
    if open_implied is None or close_implied is None:
        return None
    # Beating the close means paying a lower implied probability than the market.
    return round((close_implied - open_implied) * 100, 2)


def _prediction_matches_actual(predicted: str | None, actual: str | None) -> bool:
    if not predicted or not actual:
        return False
    if predicted == actual:
        return True
    return team_match_score(predicted, actual) >= 0.92


# A pick on a game that never happened has no result and never will. Left alone
# it sits at "pending" forever: 12 picks were stuck that way, the oldest from
# 2026-06-18, and 10 of them were rain-outs replayed later -- six as the second
# game of a doubleheader the very next day, which is what a postponed MLB game
# normally becomes.
#
# Two ways a pick reaches that dead end, so both are handled:
#   1. ESPN still lists the game on its original date, flagged postponed. Seen
#      directly, voided immediately.
#   2. ESPN drops the game from that date entirely once it is rescheduled. Never
#      seen, so it can only be aged out.
#
# The makeup game is predicted and graded in its own right, so voiding the
# original is also what stops one postponement being counted twice.
VOID_UNRESOLVED_AFTER_DAYS = 3


def _abandoned_reason(game: dict[str, Any]) -> str | None:
    """Why this game will not produce a result, or None if it still might.

    Deliberately does not include `isDelayed` -- a rain delay is a game that has
    not finished yet, not one that was called off.
    """
    if game.get("isVoided"):
        return "voided"
    if game.get("isCanceled"):
        return "canceled"
    if game.get("isWashedOut"):
        return "washed out"
    if game.get("isPostponed"):
        return "postponed"
    return None


def grade_total(pick: dict[str, Any] | None, home_score: Any, away_score: Any) -> dict[str, Any] | None:
    """Score an over/under pick against the final total.

    A total landing exactly on the line is a push -- the stake comes back. That
    is neither a win nor a loss, and counting it either way misstates the record,
    so it gets its own outcome.
    """
    if not pick:
        return None
    try:
        line = float(pick["line"])
        actual = int(home_score) + int(away_score)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    side = str(pick.get("pickSide") or "").lower()
    if side not in {"over", "under"}:
        return None

    if actual == line:
        outcome = "push"
    elif (actual > line) == (side == "over"):
        outcome = "win"
    else:
        outcome = "loss"

    # None whenever the pick has no logged price -- SBR carries no odds for
    # this market at all, so most graded totals will not have one yet. A push
    # returns its stake, not a profit or a loss, so it never prices a unit.
    odds = pick.get("odds")
    units = (
        american_odds_profit(odds, outcome == "win")
        if odds is not None and outcome != "push"
        else None
    )

    return {
        "line": line,
        "pickSide": side,
        "actual": actual,
        "outcome": outcome,
        "odds": odds,
        "units": units,
    }


def grade_spread(
    pick: dict[str, Any] | None, home_score: Any, away_score: Any
) -> dict[str, Any] | None:
    """Score a spread or runline pick against the final margin.

    `line` is the home side's number, so a home favourite carries a negative
    one. The home side covers when its margin beats that number; landing exactly
    on it is a push, which cannot happen on a half-point line like baseball's
    -1.5 but can on a whole-number spread.
    """
    if not pick:
        return None
    try:
        line = float(pick["line"])
        margin = int(home_score) - int(away_score)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    side = str(pick.get("pickSide") or "").lower()
    if side not in {"home", "away"}:
        # "push"/"No lean" picks are not positions and are not graded.
        return None

    adjusted = margin + line
    if adjusted == 0:
        outcome = "push"
    elif (adjusted > 0) == (side == "home"):
        outcome = "win"
    else:
        outcome = "loss"

    odds = pick.get("odds")
    units = (
        american_odds_profit(odds, outcome == "win")
        if odds is not None and outcome != "push"
        else None
    )

    return {
        "line": line,
        "pickSide": side,
        "margin": margin,
        "market": pick.get("market") or "spread",
        "outcome": outcome,
        "odds": odds,
        "units": units,
    }


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


def _clv_block(values: list[float]) -> dict[str, Any]:
    """Rate, median and mean for one population of confirmed CLV readings.

    Reports both tails of the coin-flip test. There used to be only
    `beatsCoinFlip`, which asks whether the model is provably good and answers
    False for "provably bad" and "too thin to say" alike -- so a rate two
    standard errors on the wrong side read exactly like no evidence at all.
    That is how 38.9% over 90 picks sat unremarked next to a +2.6% return.
    """
    if not values:
        return {
            "picks": 0, "medianPct": None, "avgPct": None, "beatCloseP": None,
            "beatCloseStdErrPct": None, "beatsCoinFlip": None,
            "worseThanCoinFlip": None, "unmoved": 0,
        }
    beat = sum(1 for value in values if value > 0)
    # A line that never moved is not a loss. Only 3 picks in 90 as measured, so
    # it does not drive the headline -- but counting a non-event as a defeat is
    # the kind of quiet bias that is much harder to find later than now.
    unmoved = sum(1 for value in values if value == 0)
    rate = beat / len(values) * 100
    std_err = math.sqrt(0.25 / len(values)) * 100
    return {
        "picks": len(values),
        # Median first, deliberately. The mean is dragged toward zero by a
        # handful of large favourable moves: over the full record it reads
        # -0.16% against a median of -0.61%, which is the difference between
        # "roughly break-even" and "consistently the wrong side of the close".
        "medianPct": round(statistics.median(values), 2),
        "avgPct": round(statistics.fmean(values), 2),
        "beatCloseP": round(rate, 1),
        "beatCloseStdErrPct": round(std_err, 1),
        "beatsCoinFlip": rate - 1.96 * std_err > 50.0,
        "worseThanCoinFlip": rate + 1.96 * std_err < 50.0,
        "unmoved": unmoved,
    }


def clv_summary(
    all_results: list[dict[str, Any]], recent: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Closing line value over picks that actually have a closing line.

    Only picks whose price was frozen at kick-off have one. The rest carry the
    latest quote seen, which is not a close, and averaging the two together
    produced a headline that could not be acted on.

    Scored over the **whole** confirmed record rather than the recent window.
    The two disagree in a way that matters: measured 12 Aug the window read
    42.3% over 71 picks, an interval that spans 50 and licenses "not
    conclusive", while the full 90 read 38.9% with an interval of 28.5-49.3
    that does not. The window is still reported, under `last7Days`, because a
    trend is worth seeing -- it is just not the headline.

    Split by league for the same reason the market records are split: pooling
    hides that MLB (35.4%, n=65) and WNBA (56.2%, n=16) point opposite ways,
    and only one of them has the sample to mean anything.
    """
    def _confirmed(rows: list[dict[str, Any]]) -> list[float]:
        return [
            float(item["clvPct"])
            for item in rows
            if item.get("clvPct") is not None and item.get("pickOddsFrozenAt")
        ]

    confirmed = _confirmed(all_results)
    summary = _clv_block(confirmed)

    by_league: dict[str, dict[str, Any]] = {}
    leagues = {item.get("league") or "unknown" for item in all_results}
    for league in sorted(leagues):
        values = _confirmed([r for r in all_results if (r.get("league") or "unknown") == league])
        if values:
            by_league[league] = _clv_block(values)
    summary["byLeague"] = by_league

    if recent is not None:
        summary["last7Days"] = _clv_block(_confirmed(recent))

    # Picks with a price but no confirmed close yet. Excluded above rather than
    # silently mixed in.
    summary["provisionalPicks"] = sum(
        1
        for item in all_results
        if item.get("clvPct") is not None and not item.get("pickOddsFrozenAt")
    )
    summary["note"] = (
        "Closing line value predicts long-run profitability better than win rate. "
        "Measured only on picks whose price was frozen when the game started, over "
        "the whole graded record rather than the recent window. Median is the "
        "headline figure; the mean is pulled by a few large favourable moves."
    )
    return summary


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

    # Only dates we actually managed to read can age a pick out. A pick must
    # never be voided because ESPN happened to be down when we looked.
    checked_dates: set[tuple[str, str]] = set()
    abandoned_events: dict[str, str] = {}

    for league, check_date in sorted(dates_to_check):
        try:
            scoreboard = fetch_scoreboard(league, check_date, retries=2, retry_delay=0.5, verify_ssl=verify_ssl)
            games = parse_scoreboard(scoreboard, league=league)
        except Exception as exc:
            skipped_dates.append({"league": league, "date": check_date, "error": str(exc)})
            continue

        checked_dates.add((league, check_date))

        for game in games:
            event_id = str(game.get("eventId") or "")
            if not event_id or event_id in graded_ids:
                continue
            pending = log.get("predictions", {}).get(event_id)
            if not pending:
                continue
            reason = _abandoned_reason(game)
            if reason:
                abandoned_events[event_id] = reason
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

    # Close out picks that can no longer produce a result. Without this they stay
    # "pending" forever and the pending count stops meaning "not played yet".
    for event_id, record in picks_by_event.items():
        if record.get("status") != "pending":
            continue

        reason = abandoned_events.get(event_id)
        if reason:
            record["status"] = "voided"
            record["voidReason"] = reason
            continue

        league = record.get("league")
        schedule_date = record.get("scheduleDate")
        if not league or not schedule_date:
            continue
        # Rescheduled games vanish from their original date, so they can only be
        # aged out -- and only against a date we actually read this run.
        if (league, schedule_date) not in checked_dates:
            continue
        try:
            age_days = (
                date.fromisoformat(league_schedule_date(league, 0))
                - date.fromisoformat(schedule_date)
            ).days
        except ValueError:
            continue
        if age_days > VOID_UNRESOLVED_AFTER_DAYS:
            record["status"] = "voided"
            record["voidReason"] = "no result reported"

    # Reconcile `published` against the log on every run. Records already graded
    # are never rebuilt, so without this a row keeps whatever flag it had when it
    # was first written -- and every row from before the publish/log split has no
    # flag at all, which defaults to published. A threshold change would then only
    # apply to picks made after it, silently leaving the old ones in the record.
    for event_id, record in picks_by_event.items():
        pending = predictions.get(event_id)
        if pending is None:
            continue
        record["published"] = bool(pending.get("published", True))

        # Side markets need the same treatment, and for the same reason. Eight
        # games were graded with a total sitting in the log and no result against
        # it, because they had already graded before side-market scoring shipped
        # and an already-graded record is never rebuilt. Without this the totals
        # record would have started from whatever graded next and silently
        # discarded every earlier pick.
        if record.get("status") != "graded":
            continue
        home_score, away_score = record.get("homeScore"), record.get("awayScore")
        if home_score is None or away_score is None:
            continue
        if not record.get("totalResult"):
            record["totalResult"] = grade_total(pending.get("total"), home_score, away_score)
        if not record.get("spreadResult"):
            record["spreadResult"] = grade_spread(pending.get("spread"), home_score, away_score)

    # Accuracy and ROI describe what the board actually showed. Picks the
    # threshold withheld are kept in the log so the model can still learn from
    # them, but counting them here would report a record nobody could have bet.
    all_results = sorted(
        (item for item in picks_by_event.values() if item.get("published", True)),
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
        elif item.get("status") == "voided":
            # Reported rather than dropped. A void is a real outcome -- the game
            # was called off -- and silently discarding it would make picks
            # disappear from the count with no explanation.
            all_time["voided"] = all_time.get("voided", 0) + 1

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

    # Every pick with a frozen close, graded or not, with the recent graded
    # window alongside. CLV is a fact about the price, not about the result --
    # a pick whose close was captured at first pitch has a measurable CLV the
    # moment the game starts, and waiting for the outcome only delays the one
    # metric that is supposed to read faster than realised return.
    clv_summary_payload = clv_summary(list(picks_by_event.values()), window)

    streak = _compute_streak(recent_results)

    accuracy = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "last7Days": last7,
            "allTime": all_time,
            "byLeague": by_league,
            "streak": streak,
            "closingLineValue": clv_summary_payload,
            # Their own buckets, never folded into the moneyline record above.
            # A good totals week must not flatter a bad picking week.
            "totals": _market_summary(all_results, "totalResult"),
            "spreads": _market_summary(all_results, "spreadResult"),
        },
        "recentResults": recent_results,
        "pendingPicks": pending_picks,
        "picksByEventId": picks_by_event,
        "skippedDates": skipped_dates,
    }
    _save_json(accuracy_path, accuracy)
    return accuracy
