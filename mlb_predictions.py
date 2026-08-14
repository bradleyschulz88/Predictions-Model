"""Win probability model and human-readable reasoning for scheduled games."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

from sports_config import get_league

from calibration_params import (
    calibrate_probability as _calibrate_probability,
    is_publishable_pick,
    load_calibration_params,
)

from data_providers.league_metrics import (
    league_metrics_logit_adjustment,
    soccer_draw_probability,
)
from data_providers.mlb_pitcher import mlb_pitching_logit_adjustment
from data_providers.park_factors import park_run_environment
from data_providers.matchup_context import handedness_diff
from data_providers.travel import travel_edge
from data_providers.schedule_advanced import schedule_flags_logit_adjustment
from data_providers.enrich import enrich_games_with_providers
from elo import load_ratings, rating_edge
from market import american_to_decimal, assess_price, devig_power, devig_proportional
from youtube_intel import intel_edge, load_intel
from model_core import resolve_probabilities
from shared_utils import (
    games_played,
    parse_record,
    win_pct_from_record,
    win_pct_or_none,
    format_win_pct,
)

HOME_FIELD_LOGIT = {
    "mlb": 0.28,
    "nfl": 0.32,
    "nba": 0.24,
    "wnba": 0.24,
    "epl": 0.30,
    "afl": 0.22,
}

DEFAULT_DRAW_PROB = 0.26

MARKET_BLEND_WEIGHT = {
    "mlb": 0.10,
    "nfl": 0.15,
    "nba": 0.15,
    "wnba": 0.12,
    "epl": 0.12,
    "afl": 0.06,
}
DEFAULT_MARKET_BLEND_WEIGHT = 0.10

_CALIBRATION_PARAMS: dict[str, Any] | None = None
_EVALUATION_REPORT: dict[str, Any] | None = None

# Below this many graded picks at a confidence level, evaluation.py itself
# calls the reliability bucket noise rather than signal -- reuse the same
# floor here rather than sizing a stake off a bucket of four games.
MIN_KELLY_BAND_SAMPLE = 30


def american_odds_to_implied(odds: int | float) -> float:
    value = float(odds)
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    return 100.0 / (value + 100.0)


def _line_odds_value(line: dict[str, Any], *keys: str) -> int | float | None:
    """Read American odds from SBR (homeOdds) or ESPN (home) line shapes."""
    for key in keys:
        value = line.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)) and value == 0:
            continue
        if isinstance(value, str):
            text = value.strip().replace("+", "")
            try:
                return int(text)
            except ValueError:
                continue
        if isinstance(value, (int, float)):
            return value
    return None


def _moneyline_from_line(line: dict[str, Any]) -> dict[str, Any] | None:
    view_type = line.get("viewType") or ""
    if "MoneyLine" not in view_type:
        return None
    current = line.get("currentLine") or line.get("openingLine")
    if not isinstance(current, dict):
        return None
    home_ml = _line_odds_value(current, "home", "homeOdds")
    away_ml = _line_odds_value(current, "away", "awayOdds")
    if home_ml is None or away_ml is None:
        return None
    draw_ml = _line_odds_value(current, "draw", "drawOdds")
    raw_home = american_odds_to_implied(home_ml)
    raw_away = american_odds_to_implied(away_ml)
    raw_draw = american_odds_to_implied(draw_ml) if draw_ml is not None else 0.0
    raw_total = raw_home + raw_away + raw_draw
    if raw_total <= 0:
        return None

    # Books load more of their margin onto the longshot, so dividing the
    # overround out evenly leaves the favourite understated. The power method
    # solves for the exponent that makes the prices sum to one, which corrects
    # that skew -- worth about a point on a lopsided market and nothing at all
    # on a pick'em. The proportional figure is kept alongside so the change can
    # be measured rather than assumed.
    sides = [raw_home, raw_away] + ([raw_draw] if raw_draw else [])
    fair = devig_power(sides)
    proportional = devig_proportional(sides)

    return {
        "sportsbook": line.get("sportsbook") or "Unknown",
        "homeOdds": home_ml,
        "awayOdds": away_ml,
        "drawOdds": draw_ml,
        "raw": {
            "home": round(raw_home * 100, 2),
            "away": round(raw_away * 100, 2),
            "draw": round(raw_draw * 100, 2) if raw_draw else None,
            "vigPct": round(max(0.0, raw_total - 1.0) * 100, 2),
        },
        "devigged": {
            "home": fair[0],
            "away": fair[1],
            "draw": fair[2] if raw_draw else None,
            "method": "power",
        },
        "devigProportional": {"home": proportional[0], "away": proportional[1]},
    }


def extract_moneyline_probs(
    lines: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    implied = compute_implied_probabilities(lines)
    if not implied.get("available"):
        return None, None, None
    consensus = implied["consensus"]
    return consensus.get("home"), consensus.get("away"), consensus.get("draw")


def has_moneyline_lines(lines: list[dict[str, Any]]) -> bool:
    for line in lines or []:
        if _moneyline_from_line(line):
            return True
    return False


def compute_implied_probabilities(lines: list[dict[str, Any]]) -> dict[str, Any]:
    books: list[dict[str, Any]] = []
    for line in lines or []:
        parsed = _moneyline_from_line(line)
        if parsed:
            books.append(parsed)

    if not books:
        return {"available": False, "booksUsed": 0, "books": [], "consensus": None}

    home_avg = sum(book["devigged"]["home"] for book in books) / len(books)
    away_avg = sum(book["devigged"]["away"] for book in books) / len(books)
    draw_values = [book["devigged"]["draw"] for book in books if book["devigged"]["draw"] is not None]
    draw_avg = sum(draw_values) / len(draw_values) if draw_values else None

    if draw_avg is not None:
        total = home_avg + away_avg + draw_avg
        if total > 0:
            home_avg /= total
            away_avg /= total
            draw_avg /= total

    raw_home_avg = sum(book["raw"]["home"] for book in books) / len(books)
    raw_away_avg = sum(book["raw"]["away"] for book in books) / len(books)
    vig_avg = sum(book["raw"]["vigPct"] for book in books) / len(books)

    return {
        "available": True,
        "booksUsed": len(books),
        "books": [
            {
                "sportsbook": book["sportsbook"],
                "homePct": round(book["devigged"]["home"] * 100, 1),
                "awayPct": round(book["devigged"]["away"] * 100, 1),
                "drawPct": round(book["devigged"]["draw"] * 100, 1) if book["devigged"]["draw"] is not None else None,
                "vigPct": book["raw"]["vigPct"],
            }
            for book in books
        ],
        "consensus": {
            "home": home_avg,
            "away": away_avg,
            "draw": draw_avg,
            "homePct": round(home_avg * 100, 1),
            "awayPct": round(away_avg * 100, 1),
            "drawPct": round(draw_avg * 100, 1) if draw_avg is not None else None,
            "rawHomePct": round(raw_home_avg, 1),
            "rawAwayPct": round(raw_away_avg, 1),
            "avgVigPct": round(vig_avg, 2),
        },
    }


def compute_true_probabilities(
    *,
    model_home: float,
    enrichment: dict[str, Any],
    league_config: Any,
    league: str = "mlb",
    lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = [
        {
            "source": "Analytics model",
            "detail": "Records, splits, pitching, form, injuries, rest, H2H, advanced stats",
            "home": model_home,
            "weight": 1.0,
        }
    ]

    espn_home = enrichment.get("espnPredictorHome")
    espn_away = enrichment.get("espnPredictorAway")
    if espn_home is not None and espn_away is not None:
        espn_total = espn_home + espn_away
        if espn_total > 0:
            components.append(
                {
                    "source": "ESPN Matchup Predictor",
                    "detail": f"{espn_home:.1f}% / {espn_away:.1f}%",
                    "home": espn_home / espn_total,
                    "weight": 0.35,
                }
            )

    weight_total = sum(item["weight"] for item in components)
    home_true = sum(item["home"] * item["weight"] for item in components) / weight_total
    away_true = 1.0 - home_true

    implied = compute_implied_probabilities(lines or [])
    if implied.get("available"):
        market_home = implied["consensus"]["home"]
        market_away = implied["consensus"]["away"]
        market_weight = MARKET_BLEND_WEIGHT.get(league, DEFAULT_MARKET_BLEND_WEIGHT)
        home_true = home_true * (1.0 - market_weight) + market_home * market_weight
        away_true = away_true * (1.0 - market_weight) + market_away * market_weight
        total = home_true + away_true
        if total > 0:
            home_true /= total
            away_true /= total

    home_true = clamp(home_true)
    away_true = clamp(away_true)

    draw_true = 0.0
    if league_config.supports_draw:
        draw_true = clamp(
            soccer_draw_probability(
                league=league,
                home_true=home_true,
                away_true=away_true,
                enrichment=enrichment,
            ),
            0.08,
            0.32,
        )
        scale = 1.0 - draw_true
        home_true *= scale
        away_true *= scale

    return {
        "home": home_true,
        "away": away_true,
        "draw": draw_true if league_config.supports_draw else None,
        "homePct": round(home_true * 100, 1),
        "awayPct": round(away_true * 100, 1),
        "drawPct": round(draw_true * 100, 1) if league_config.supports_draw else None,
        "components": [
            {
                "source": item["source"],
                "detail": item["detail"],
                "homePct": round(item["home"] * 100, 1),
                "weightPct": round(item["weight"] / weight_total * 100, 1),
            }
            for item in components
        ],
    }


def _build_team_probabilities(
    *,
    true_probs: dict[str, Any],
    implied_probs: dict[str, Any],
    blended: dict[str, Any],
) -> dict[str, Any]:
    consensus = implied_probs.get("consensus") or {}
    available = bool(implied_probs.get("available"))

    def side_block(side: str, true_key: str, implied_key: str, blended_key: str) -> dict[str, Any]:
        true_pct = true_probs.get(true_key)
        implied_pct = consensus.get(implied_key) if available else None
        blended_pct = blended.get(blended_key)
        edge_pct = None
        if true_pct is not None and implied_pct is not None:
            edge_pct = round(true_pct - implied_pct, 1)
        return {
            "truePct": true_pct,
            "impliedPct": implied_pct,
            "blendedPct": blended_pct,
            "edgePct": edge_pct,
            "edgeLabel": f"{edge_pct:+.1f}%" if edge_pct is not None else None,
        }

    teams: dict[str, Any] = {
        "home": side_block("home", "homePct", "homePct", "homePct"),
        "away": side_block("away", "awayPct", "awayPct", "awayPct"),
    }
    if true_probs.get("drawPct") is not None:
        teams["draw"] = side_block("draw", "drawPct", "drawPct", "drawPct")
    return teams


def _probability_edge(
    *,
    predicted_side: str,
    true_home: float,
    true_away: float,
    implied_home: float | None,
    implied_away: float | None,
) -> dict[str, Any] | None:
    if implied_home is None or implied_away is None or predicted_side not in {"home", "away"}:
        return None
    true_side = true_home if predicted_side == "home" else true_away
    implied_side = implied_home if predicted_side == "home" else implied_away
    edge = (true_side - implied_side) * 100
    return {
        "truePct": round(true_side * 100, 1),
        "impliedPct": round(implied_side * 100, 1),
        "edgePct": round(edge, 1),
        "edgeLabel": f"{edge:+.1f}% true vs implied",
        "favorsModel": abs(edge) >= 3,
        "modelPct": round(true_side * 100, 1),
        "marketPct": round(implied_side * 100, 1),
    }


def compute_total_implied_probabilities(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    overs: list[float] = []
    unders: list[float] = []
    for line in lines or []:
        if "Total" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        over_odds = _line_odds_value(current, "over", "overOdds")
        under_odds = _line_odds_value(current, "under", "underOdds")
        if over_odds is None or under_odds is None:
            continue
        if isinstance(over_odds, str):
            over_match = re.search(r"\(([+-]?\d+)\)", over_odds)
            under_match = re.search(r"\(([+-]?\d+)\)", under_odds)
            if not over_match or not under_match:
                continue
            over_imp = american_odds_to_implied(int(over_match.group(1)))
            under_imp = american_odds_to_implied(int(under_match.group(1)))
        else:
            over_imp = american_odds_to_implied(over_odds)
            under_imp = american_odds_to_implied(under_odds)
        total = over_imp + under_imp
        if total <= 0:
            continue
        overs.append(over_imp / total)
        unders.append(under_imp / total)

    if not overs:
        return None
    over_avg = sum(overs) / len(overs)
    under_avg = sum(unders) / len(unders)
    return {
        "overPct": round(over_avg * 100, 1),
        "underPct": round(under_avg * 100, 1),
        "booksUsed": len(overs),
    }


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp(value: float, low: float = 0.05, high: float = 0.95) -> float:
    return max(low, min(high, value))


def _format_plus_minus(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return f"{int(number):+d}"
    return f"{number:+.1f}"


def _edge_label(home_value: float, away_value: float) -> str:
    if abs(home_value - away_value) < 0.01:
        return "even"
    return "home" if home_value > away_value else "away"


def _team_by_side(game: dict[str, Any], side: str) -> str | None:
    return game.get("homeTeam") if side == "home" else game.get("awayTeam")


def _last_five_pct(record: str | None, league: str | None = None) -> float | None:
    """Win rate over the last five games, or None when it cannot be read.

    The previous sentinel (-1.0) was indistinguishable from a real win rate once
    it reached the logit, so an unreadable or 0-0 record now returns None and the
    form term is skipped entirely.
    """
    if not record or not parse_record(record):
        return None
    pct = win_pct_from_record(record, default=-1.0, league=league)
    return None if pct < 0 else pct


def _league_id(game: dict[str, Any]) -> str:
    return game.get("league") or "mlb"


def extract_total_line(lines: list[dict[str, Any]]) -> float | None:
    for line in lines:
        if "Total" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        for side in ("over", "under"):
            value = current.get(side)
            if not value:
                continue
            text = str(value).lower().lstrip("ou")
            try:
                return float(text.split()[0].replace("o", "").replace("u", ""))
            except ValueError:
                continue
    return None


def extract_spread_line(lines: list[dict[str, Any]]) -> float | None:
    """Extract point spread line from odds data. Returns home team spread (negative = home favorite)."""
    for line in lines:
        if "Spread" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue

        # Read the home spread directly when present. Falling through to the away
        # field (as this used to) returns the opposite side's number under the
        # home sign convention, flipping the favourite.
        for key, sign in (("home", 1.0), ("away", -1.0)):
            value = current.get(key)
            if not value:
                continue
            text = str(value).replace("+", "").replace("−", "-").replace("–", "-")
            try:
                return float(text.split()[0]) * sign
            except ValueError:
                continue
    return None


# SBR's total/spread lines are bare numbers with no price attached -- there is
# nowhere on that source for one to live. ESPN's core odds embed it as
# parenthetical text alongside the line itself, e.g. "o8.5 (-110)" or
# "+1.5 (-108)" (see espn_odds.py's _market_lines), because that endpoint has
# no separate field for it either. Both extractors below return None on an
# SBR-sourced line, which is correct: the price genuinely is not there.
_PRICE_IN_PARENS = re.compile(r"\(([+-]\d+)\)")


def extract_total_price(lines: list[dict[str, Any]], side: str) -> int | None:
    """American price for the over/under side actually picked, if logged."""
    if side not in ("over", "under"):
        return None
    for line in lines:
        if "Total" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        value = current.get(side)
        if value is None:
            continue
        match = _PRICE_IN_PARENS.search(str(value))
        if match:
            return int(match.group(1))
    return None


def extract_spread_price(lines: list[dict[str, Any]], side: str) -> int | None:
    """American price for the spread side actually picked, if logged."""
    if side not in ("home", "away"):
        return None
    for line in lines:
        if "Spread" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        value = current.get(side)
        if value is None:
            continue
        match = _PRICE_IN_PARENS.search(str(value))
        if match:
            return int(match.group(1))
    return None


# A side market may only be promoted to a game's headline bet once it has this
# many PRICED graded picks behind it. Priced, not merely graded: an unpriced
# pick contributes no return, so it is no evidence that betting the market pays.
# 30 is the bar this codebase already uses for "enough to conclude from"
# (evaluation.MIN_RELIABLE_SAMPLE, MIN_KELLY_BAND_SAMPLE, MIN_LEAGUE_HISTORY).
#
# Counted per market across all leagues rather than per market per league:
# split six ways the side-market record is under ten picks everywhere, so a
# per-league bar would never open and the gate would just be an off switch.
MIN_MARKET_HISTORY = 30

_ACCURACY_REPORT: dict[str, Any] | None = None


def _get_accuracy_report() -> dict[str, Any]:
    """docs/data/accuracy.json -- the graded record, for gating decisions.

    Read the same way the evaluation report is, and absent is the normal case
    in a fresh checkout or in tests. An absent record gates every side market
    off, which is the safe direction: it means "no evidence yet", not "fine".
    """
    global _ACCURACY_REPORT
    if _ACCURACY_REPORT is None:
        path = Path(__file__).resolve().parent / "docs" / "data" / "accuracy.json"
        try:
            _ACCURACY_REPORT = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _ACCURACY_REPORT = {}
    return _ACCURACY_REPORT


# prediction key -> the accuracy summary block that records how it has done.
_MARKET_RECORD_KEY = {"total": "totals", "spread": "spreads"}


def market_record(market: str) -> dict[str, Any]:
    """This side market's graded record, as accuracy.json summarised it."""
    key = _MARKET_RECORD_KEY.get(market)
    if key is None:
        return {}
    return (_get_accuracy_report().get("summary") or {}).get(key) or {}


def market_priced_history(market: str) -> int:
    """How many priced graded picks this side market has behind it."""
    priced = market_record(market).get("priced")
    return int(priced) if isinstance(priced, (int, float)) else 0


def market_is_validated(market: str) -> bool:
    """Whether this market has earned the right to headline a game.

    The moneyline always has: it is the fitted, calibrated output the whole
    model is built around. The side markets are heuristics -- a stack of
    hand-tuned leans for totals, a normal margin model for spreads -- so they
    have to show a record first.

    Two conditions, because they answer different questions. Enough PRICED
    picks says the market is actually bettable and its return is measurable;
    a priced hit rate clearing the break-even those prices imply, by more than
    its own error bar, says the record actually points the right way.

    Both halves of that were wrong before, and in the same direction.

    It compared `pct`, the hit rate over EVERY graded pick, against a
    break-even that only applies to the PRICED ones -- the same blended-versus-
    priced confusion already fixed in the accuracy report, still sitting in the
    one place that decides whether to stake real money. Measured 13 Aug it was
    live: spreads read 57.1% blended and passed, while the priced record it
    would actually be staked at was 52.3% against a 53.4% break-even. The gate
    was backing a market on the strength of picks that carried no price.

    And it compared on the point estimate alone. Totals cleared its break-even
    by a tenth of a percentage point -- 52.3 against 52.2, on a standard error
    near four -- which is not evidence of anything. Requiring one standard
    error of daylight is not a significance test, which would need roughly two;
    it is the weaker claim that the record has some room to spare rather than
    sitting on the line.

    An earlier version of this docstring argued against any such margin on the
    grounds that no side market would ever be backed. That is now the correct
    outcome rather than an over-strict one: neither side market has a priced
    record above its break-even, so backing either would be staking on a
    number that says nothing. They stay ranked, priced and visible on the card
    -- publish-only, not hidden -- and the gate reopens on its own if a record
    earns it.
    """
    if market == "moneyline":
        return True
    record = market_record(market)
    if market_priced_history(market) < MIN_MARKET_HISTORY:
        return False
    pct = record.get("pricedPct")
    std_err = record.get("pricedStdErrPct")
    break_even = record.get("breakEvenPct")
    if not all(isinstance(value, (int, float)) for value in (pct, std_err, break_even)):
        return False
    return pct - std_err > break_even


def _get_calibration_params() -> dict[str, Any]:
    global _CALIBRATION_PARAMS
    if _CALIBRATION_PARAMS is None:
        _CALIBRATION_PARAMS = load_calibration_params()
    return _CALIBRATION_PARAMS


def _get_evaluation_report() -> dict[str, Any]:
    """docs/data/evaluation.json, if the build already wrote one this run.

    Refit happens before this module's predictions do (see pages.yml), so the
    reliability curve on disk reflects every graded game up to today, not the
    picks about to be made -- there is no leakage. Missing or unparsable is
    the normal case in a fresh checkout or in tests, and is silently empty.
    """
    global _EVALUATION_REPORT
    if _EVALUATION_REPORT is None:
        path = Path(__file__).resolve().parent / "docs" / "data" / "evaluation.json"
        try:
            _EVALUATION_REPORT = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _EVALUATION_REPORT = {}
    return _EVALUATION_REPORT


def kelly_band_probability(confidence_pct: float | None) -> float | None:
    """The measured win rate of picks this confident, for sizing a stake.

    A model can be miscalibrated even after the shrinkage calibrate_probability
    already applies -- the 85-90% band has, at times, won barely half its
    games. Kelly staked on the model's own 85-90% number treats that gap as
    real edge instead of the overconfidence it is. When a confidence band has
    enough graded history to trust (>= MIN_KELLY_BAND_SAMPLE picks), its
    actual win rate is a better estimate of the true probability than the
    number the model asserts, so the caller should size against this instead.
    Returns None -- "use the model's own number" -- whenever there isn't
    enough history yet, so an early or thin build changes nothing.
    """
    if confidence_pct is None or confidence_pct < 50:
        return None
    reliability = _get_evaluation_report().get("reliability") or []
    for bucket in reliability:
        try:
            lo_str, hi_str = str(bucket.get("range", "")).split("-")
            lo, hi = float(lo_str), float(hi_str)
        except (ValueError, AttributeError):
            continue
        if not (lo <= confidence_pct < hi):
            continue
        picks = bucket.get("picks")
        actual = bucket.get("actualWinPct")
        if (
            isinstance(picks, (int, float))
            and picks >= MIN_KELLY_BAND_SAMPLE
            and isinstance(actual, (int, float))
        ):
            return max(0.0, min(1.0, actual / 100.0))
        return None
    return None


def calibrate_probability(
    prob: float,
    *,
    league: str = "mlb",
    confidence_pct: float | None = None,
) -> float:
    """Pull extreme probabilities toward 50% using graded calibration buckets."""
    return _calibrate_probability(
        prob,
        league=league,
        confidence_pct=confidence_pct,
        params=_get_calibration_params(),
    )


def confidence_label(confidence_pct: float) -> str:
    if confidence_pct >= 68:
        return "Strong pick"
    if confidence_pct >= 57:
        return "Lean"
    return "Coin flip"


def _injury_role_weight(injury: dict[str, Any], league: str) -> float:
    detail = f"{injury.get('player', '')} {injury.get('detail', '')} {injury.get('status', '')}".lower()
    weight = 1.0
    if league == "nfl" and any(token in detail for token in ("quarterback", " qb")):
        weight = 2.5
    elif league == "mlb" and "pitcher" in detail:
        weight = 2.0
    elif league in {"nba", "wnba"} and any(token in detail for token in ("out", "doubtful")):
        weight = 1.4
    status = (injury.get("status") or "").lower()
    if any(token in status for token in ("out", "il", "suspended")):
        weight *= 1.15
    return weight


def _weighted_injury_score(injuries: list[dict[str, Any]], league: str) -> float:
    return sum(_injury_role_weight(injury, league) for injury in injuries)


def _injury_logit_adjustment(enrichment: dict[str, Any], league: str = "mlb") -> float:
    home_load = _weighted_injury_score(enrichment.get("homeMajorInjuries") or [], league)
    away_load = _weighted_injury_score(enrichment.get("awayMajorInjuries") or [], league)
    return (away_load - home_load) * 0.18


def _streak_score(profile: dict[str, Any]) -> float:
    streak_type = profile.get("streakType")
    streak_num = profile.get("streakNumber")
    if streak_num is None or not streak_type:
        return 0.0
    sign = 1.0 if str(streak_type).lower() == "win" else -1.0
    return sign * min(5.0, float(streak_num)) * 0.04


def _streak_logit_adjustment(enrichment: dict[str, Any]) -> float:
    home = _streak_score(enrichment.get("homeAdvanced") or {})
    away = _streak_score(enrichment.get("awayAdvanced") or {})
    return max(-0.25, min(0.25, home - away))


def _parse_batting_avg(stat_line: str | None) -> float | None:
    if not stat_line:
        return None
    match = re.search(r"(\.\d{3})", stat_line)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


# Each lineup metric lives on its own scale, so a raw home-minus-away subtraction
# is only meaningful when both sides were scored the same way. These centres and
# spreads normalise each metric to roughly mean 0 / unit 1 before comparison.
_LINEUP_METRIC_SCALE = {
    # metric: (league-average value, spread of one "unit" of edge)
    "battingAvg": (0.250, 0.020),
    "opsProxy": (0.720, 0.050),
    "pointsPerGame": (110.0, 8.0),
}


def _lineup_quality_score(
    game: dict[str, Any],
    side: str,
    league: str,
    enrichment: dict[str, Any],
) -> tuple[str, float] | None:
    """Return (metric_name, normalised score) so callers can refuse to mix scales.

    Previously this returned a bare float that could be a batting average (~0.25),
    an OPS proxy (~0.72), points-per-game/100 (~1.1) or a roster-completeness
    ratio (~1.0) depending purely on which data happened to be present. Comparing
    home to away across two different metrics manufactured edges large enough to
    saturate the clamp whenever data coverage was asymmetric.
    """
    lineup = game.get(f"{side}Lineup") or {}
    batters = lineup.get("batters") or []
    advanced = enrichment.get("homeAdvanced" if side == "home" else "awayAdvanced") or {}

    def normalise(metric: str, value: float) -> tuple[str, float]:
        centre, spread = _LINEUP_METRIC_SCALE[metric]
        return metric, (float(value) - centre) / spread

    if batters and league == "mlb":
        confirmed = [batter for batter in batters if batter.get("order")] or batters
        averages: list[float] = []
        for batter in confirmed:
            avg = batter.get("avg")
            if avg is None and batter.get("statLine"):
                avg = _parse_batting_avg(batter.get("statLine"))
            if avg is not None:
                try:
                    averages.append(float(avg))
                except (TypeError, ValueError):
                    continue
        if averages:
            return normalise("battingAvg", sum(averages) / len(averages))

    if league == "mlb":
        ops = advanced.get("opsProxy")
        if ops is not None:
            return normalise("opsProxy", ops)

    if league in {"nba", "wnba", "nfl", "afl"}:
        scoring = advanced.get("pointsPerGame")
        if scoring is not None:
            return normalise("pointsPerGame", scoring)

    return None


def _lineup_logit_adjustment(game: dict[str, Any], league: str, enrichment: dict[str, Any]) -> float:
    home_score = _lineup_quality_score(game, "home", league, enrichment)
    away_score = _lineup_quality_score(game, "away", league, enrichment)
    # Only compare like with like: if one side has confirmed batting averages and
    # the other only a season OPS proxy, the difference is an artefact of data
    # coverage rather than a real edge.
    if home_score is None or away_score is None or home_score[0] != away_score[0]:
        return 0.0
    multiplier = 0.10 if league == "mlb" else 0.08
    return max(-0.35, min(0.35, (home_score[1] - away_score[1]) * multiplier))


def _weather_win_logit_adjustment(game: dict[str, Any], enrichment: dict[str, Any], league: str) -> float:
    if league != "mlb":
        return 0.0
    venue = (game.get("venueName") or "").lower()
    if any(token in venue for token in ("dome", "roof", "tropicana", "minute maid")):
        return 0.0
    run_env = (enrichment.get("weatherImpact") or {}).get("runEnvironmentAdj") or 0.0
    return max(-0.12, min(0.12, run_env * 2.0))


_ELO_RATINGS: dict[str, Any] | None = None


def _elo_ratings() -> dict[str, Any]:
    """Load the rating table once per process."""
    global _ELO_RATINGS
    if _ELO_RATINGS is None:
        _ELO_RATINGS = load_ratings()
    return _ELO_RATINGS


_VIDEO_INTEL: dict[str, Any] | None = None


def _video_intel() -> dict[str, Any]:
    """Load the YouTube news file once per process.

    Absent on every machine that has not run youtube_intel.py, which is the
    normal case -- load_intel returns {} and the feature stays None.
    """
    global _VIDEO_INTEL
    if _VIDEO_INTEL is None:
        _VIDEO_INTEL = load_intel()
    return _VIDEO_INTEL


def _pct_diff(
    home_record: str | None, away_record: str | None, *, league: str
) -> float | None:
    """Home minus away win percentage, or None when either side has no games.

    Both sides must have played for the difference to mean anything. One club
    at 0-0 against one at 6-2 is not a -0.75 edge; it is an unknown.
    """
    home = win_pct_or_none(home_record, league=league)
    away = win_pct_or_none(away_record, league=league)
    if home is None or away is None:
        return None
    return round(home - away, 4)


# ESPN's season type for an exhibition. 2 is the regular season, 3 the
# postseason.
PRESEASON_SEASON_TYPE = 1


def extract_model_inputs(game: dict[str, Any]) -> dict[str, Any]:
    """Features the probability model reads, computed before any prediction exists.

    Split out of extract_prediction_features so the fitted model can be scored
    on a game without a circular dependency on its own output.
    """
    enrichment = game.get("enrichment") or {}
    league = _league_id(game)
    home_adv = enrichment.get("homeAdvanced") or {}
    away_adv = enrichment.get("awayAdvanced") or {}
    implied = compute_implied_probabilities(game.get("lines") or [])
    consensus = implied.get("consensus") or {} if implied.get("available") else {}
    rest_days = enrichment.get("restDays") or {}
    home_flags = enrichment.get("homeScheduleFlags") or {}
    away_flags = enrichment.get("awayScheduleFlags") or {}

    data_coverage = {
        "lineup": bool((game.get("homeLineup") or {}).get("batters") or (game.get("awayLineup") or {}).get("batters")),
        "injuries": bool(enrichment.get("homeMajorInjuries") or enrichment.get("awayMajorInjuries")),
        "espnPredictor": enrichment.get("espnPredictorHome") is not None
        and enrichment.get("espnPredictorAway") is not None,
        "advancedStats": home_adv.get("powerRating") is not None or away_adv.get("powerRating") is not None,
        "restData": rest_days.get("home") is not None and rest_days.get("away") is not None,
        "scheduleFlags": bool(home_flags or away_flags),
        "mlbPitching": bool(enrichment.get("mlbPitching")),
        "leagueMetrics": bool((enrichment.get("leagueMetrics") or {}).keys() - {"league"}),
        "impliedOdds": bool(implied.get("available")),
    }

    return {
        "league": league,
        # None, not 0.0, when either club has not played yet. At the start of a
        # season every team is 0-0, so the old 0.5 default made these read as
        # "two exactly average teams" -- and splitDiff's home-field centring
        # then turned that fabricated 0.0 into a systematic edge against the
        # home side on every opening-week game. See win_pct_or_none.
        "recordDiff": _pct_diff(
            game.get("homeRecord"), game.get("awayRecord"), league=league
        ),
        "splitDiff": _pct_diff(
            game.get("homeHomeRecord"), game.get("awayRoadRecord"), league=league
        ),
        "homePower": home_adv.get("powerRating"),
        "awayPower": away_adv.get("powerRating"),
        # How many games the strength estimate actually rests on, taken from the
        # thinner of the two clubs. Without it a 1-0 record and a 12-4 record
        # were the same input, and one preseason result carried the authority of
        # a third of a season -- measured 13 Aug on NFL preseason, where a 0.80
        # to 0.00 power gap off a single game produced 92% confidence against a
        # market at 41%.
        #
        # Preseason counts as zero regardless of games played. Starters barely
        # appear -- the average starting quarterback threw 10.1 passes across an
        # entire preseason -- so the result is decided by third-string players
        # and a coaching decision about risk, and says nothing about the clubs.
        # Reporting it as zero evidence lets one shrinkage rule handle both the
        # exhibition case and every ordinary cold start, with no second path.
        "strengthGames": (
            0
            if game.get("seasonType") == PRESEASON_SEASON_TYPE
            else min(
                games_played(game.get("homeRecord")),
                games_played(game.get("awayRecord")),
            )
        ),
        # Park run index, centred on zero. Logged even though it is expected to
        # fail the moneyline ablation -- a park inflates scoring for both teams,
        # so it should barely move who wins. It is here because the honest way
        # to find that out is to measure it, and because unlike weather it can
        # be recovered for past games from the home club alone.
        "parkEdge": (park_run_environment(game.get("homeTeam"), game.get("venueName")) or {}).get("edge"),
        # Season-series record between these two clubs. Logged as a candidate,
        # with low expectations: a season series is 3-13 games, so this is mostly
        # noise dressed as history, and whatever real signal it holds is already
        # inside the season records that feed strengthDiff. It earns a place only
        # if it beats its own absence out of sample, like everything else.
        "h2hDiff": _h2h_diff(enrichment),
        # Distance and body-clock shift carried by the visiting club. Positive
        # favours the home side, matching every other diff here. Candidate only.
        #
        # No league gate: TEAM_HOME now covers MLB, NBA, NFL and the WNBA, and
        # travel_edge already returns None for a club it does not know, so the
        # table is the limit rather than a hardcoded league check. The gate was
        # withholding this from basketball, where 82 games and back-to-backs
        # across three time zones make it matter more than in baseball.
        "travelDiff": travel_edge(game.get("homeTeam"), game.get("awayTeam")),
        # Southpaw asymmetry between the two starters. Facts only -- no platoon
        # adjustment, which would need lineup splits that confirm too late.
        "handednessDiff": handedness_diff(game) if _league_id(game) == "mlb" else None,
        # Relief innings the two bullpens have absorbed lately, home minus away.
        # Filled by enrichment when team ids resolve; absent is normal.
        "bullpenDiff": enrichment.get("bullpenDiff"),
        "homeInjuryLoad": round(_weighted_injury_score(enrichment.get("homeMajorInjuries") or [], league), 2),
        "awayInjuryLoad": round(_weighted_injury_score(enrichment.get("awayMajorInjuries") or [], league), 2),
        # Availability x seriousness (x player importance when an LLM key is
        # configured), as an alternative to counting absences.
        "homeInjurySeverity": (enrichment.get("homeInjurySeverity") or {}).get("score"),
        "awayInjurySeverity": (enrichment.get("awayInjurySeverity") or {}).get("score"),
        # Strength-of-schedule aware rating gap. Logged as an ablation
        # candidate; it does not move the probability (it loses on 685 games).
        "eloEdge": rating_edge(
            _elo_ratings(), league, game.get("homeTeam"), game.get("awayTeam")
        ),
        # Pre-game team news from subscribed YouTube channels. Also an ablation
        # candidate only. intel_edge refuses any video published at or after
        # this game's start, so a recap cannot leak the result into the feature.
        "videoIntelEdge": intel_edge(
            _video_intel(),
            league,
            game.get("homeTeam"),
            game.get("awayTeam"),
            game.get("startDate"),
        ),
        "homeRest": rest_days.get("home"),
        "awayRest": rest_days.get("away"),
        "homeBackToBack": home_flags.get("backToBack"),
        "awayBackToBack": away_flags.get("backToBack"),
        "impliedHome": consensus.get("homePct"),
        "impliedAway": consensus.get("awayPct"),
        "hasLineup": data_coverage["lineup"],
        "dataCoverage": data_coverage,
        "mlbPitching": enrichment.get("mlbPitching"),
        "leagueMetrics": enrichment.get("leagueMetrics"),
    }


def _h2h_diff(enrichment: dict[str, Any]) -> float | None:
    """Season-series edge, home minus away. None when the clubs have not met.

    Requiring both sides to resolve independently made this dead on arrival: it
    was None on all 120 rows of a real build. `series_win_pct` only resolves for
    a club actually named in ESPN's summary string, and the most common summary
    -- "Series tied 1-1" -- names neither, while "Dodgers lead series 2-1" names
    only one. So the pair almost never both resolved.

    A season series is two-sided, so one known share determines the other. The
    tied case is read straight off the score.
    """
    head_to_head = enrichment.get("headToHead") or {}
    home = head_to_head.get("homeSeriesWinPct")
    away = head_to_head.get("awaySeriesWinPct")

    if home is None and away is not None:
        home = 1.0 - float(away)
    elif away is None and home is not None:
        away = 1.0 - float(home)
    elif home is None and away is None:
        # Neither club named: a tied series still carries a real score.
        score = str(head_to_head.get("seriesScore") or "")
        match = re.search(r"(\d+)\s*-\s*(\d+)", score)
        if not match:
            return None
        left, right = int(match.group(1)), int(match.group(2))
        if left + right == 0 or left != right:
            # An uneven score with neither club named cannot be assigned a side.
            return None
        home = away = 0.5

    return round(float(home) - float(away), 4)


def extract_prediction_features(game: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Model inputs plus the prediction's own output, for logging and grading."""
    true_probs = (prediction.get("probabilities") or {}).get("true") or {}

    features = extract_model_inputs(game)
    features.update(
        {
            "trueHome": true_probs.get("homePct"),
            "trueAway": true_probs.get("awayPct"),
            "confidence": prediction.get("confidence"),
            # Pre-calibration probability, logged so calibration can be fitted on
            # raw output instead of on numbers a previous calibration already
            # shrank -- that feedback loop drove the MLB shrink factors to their
            # floor.
            "rawConfidence": prediction.get("rawConfidence"),
            "rawHomeWinPct": prediction.get("rawHomeWinPct"),
            "predictedSide": prediction.get("predictedSide"),
            "probabilityMethod": prediction.get("probabilityMethod"),
        }
    )
    return features


def _advanced_logit_adjustment(enrichment: dict[str, Any], league: str = "mlb") -> float:
    home = enrichment.get("homeAdvanced") or {}
    away = enrichment.get("awayAdvanced") or {}
    adjustment = 0.0

    home_power = home.get("powerRating")
    away_power = away.get("powerRating")
    if home_power is not None and away_power is not None:
        adjustment += (home_power - away_power) * 2.2

    home_rpg = home.get("runsPerGame")
    away_rpg = away.get("runsPerGame")
    home_rapg = home.get("runsAllowedPerGame")
    away_rapg = away.get("runsAllowedPerGame")
    if home_rpg is not None and away_rapg is not None and away_rpg is not None and home_rapg is not None:
        offense_edge = (home_rpg - away_rpg) / 2.0
        defense_edge = (away_rapg - home_rapg) / 2.0
        adjustment += (offense_edge + defense_edge) * 0.35

    home_gf = home.get("pointsPerGame")
    away_gf = away.get("pointsPerGame")
    home_ga = home.get("goalsAgainstPerGame")
    away_ga = away.get("goalsAgainstPerGame")
    if home_gf is not None and away_gf is not None and home_ga is not None and away_ga is not None:
        adjustment += ((home_gf - away_gf) + (away_ga - home_ga)) * 0.45

    # Deliberately skipped for MLB: compute_power_rating() already folds opsProxy
    # and ERA into powerRating (0.10 each), which is applied above, so adding them
    # again here would double-count. See tests/test_model_improvements.py
    # DedupeLogitTests. For other leagues these fields are baseball-only and are
    # normally absent, making this branch inert rather than harmful.
    if league != "mlb":
        home_ops = home.get("opsProxy")
        away_ops = away.get("opsProxy")
        if home_ops is not None and away_ops is not None:
            adjustment += (home_ops - away_ops) * 1.8

        home_era = home.get("era")
        away_era = away.get("era")
        if home_era is not None and away_era is not None:
            adjustment += (away_era - home_era) * 0.22

    return adjustment


def _rest_logit_adjustment(enrichment: dict[str, Any]) -> float:
    rest = enrichment.get("restDays") or {}
    home_rest = rest.get("home")
    away_rest = rest.get("away")
    if home_rest is None or away_rest is None:
        return 0.0
    return max(-0.35, min(0.35, (home_rest - away_rest) * 0.12))


def _head_to_head_logit_adjustment(enrichment: dict[str, Any]) -> float:
    h2h = enrichment.get("headToHead") or {}
    home_pct = h2h.get("homeSeriesWinPct")
    away_pct = h2h.get("awaySeriesWinPct")
    if home_pct is None or away_pct is None:
        return 0.0
    return (home_pct - away_pct) * 1.4


def _scoring_pace_from_form(enrichment: dict[str, Any]) -> float | None:
    """Average *combined* score of recent games, to compare against a total line.

    This previously appended each team's score separately and averaged over all
    of them, yielding runs per team (~4) while the caller compared it against a
    combined total (~8.5). Being 2x low made `pace >= total + 0.8` unreachable
    in every league, so the over branch was dead code and every total leaned
    under.
    """
    combined: list[float] = []
    for side in ("homeLastFive", "awayLastFive"):
        for game in (enrichment.get(side) or {}).get("games") or []:
            parts = str(game.get("score") or "").replace("-", " ").split()
            values: list[float] = []
            for part in parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue
            # Only a two-sided score tells us the combined total for that game.
            if len(values) == 2:
                combined.append(values[0] + values[1])
    if not combined:
        return None
    return sum(combined) / len(combined)


def run_environment(game: dict[str, Any], enrichment: dict[str, Any]) -> dict[str, Any] | None:
    """How much the ballpark and the forecast push a game over or under.

    Separate from `predict_total` because it is knowable without a market. A
    total pick needs a posted line to lean against; the run environment does
    not, so this is shown on every baseball card while a totals pick appears
    only where a book has priced one.

    `shift` is a probability offset centred on zero, so 0.0 means the conditions
    say nothing and the caller adds it to a 0.5 base.
    """
    if _league_id(game) != "mlb":
        return None

    shift = 0.0
    notes: list[str] = []
    park_edge = None

    park = park_run_environment(game.get("homeTeam"), game.get("venueName"))
    if park:
        park_edge = park["edge"]
        # Scaled so Coors (+15) moves the lean 0.09 and a mid-table park moves
        # almost nothing, matching how the effect actually distributes: the
        # extremes are large and uncontroversial, the middle is inside the noise.
        shift += clamp(park["edge"] * 0.006, -0.09, 0.09)
        # Only the extremes are worth saying out loud; otherwise every card
        # carries a line about the ballpark being average.
        if abs(park["edge"]) >= 3:
            notes.append(
                f"{game.get('venueName') or 'The ballpark'} is {park['note']} "
                f"({park['factor']:.0f} runs index)."
            )

    weather_impact = enrichment.get("weatherImpact") or {}
    weather_adj = weather_impact.get("runEnvironmentAdj") or 0.0
    if weather_adj:
        shift += weather_adj
        if weather_impact.get("summary"):
            notes.append(f"Weather: {weather_impact['summary']}.")

    if park is None and not weather_adj:
        return None

    shift = clamp(shift, -0.18, 0.18)
    over_pct = round((0.5 + shift) * 100, 1)
    return {
        "shift": shift,
        "overPct": over_pct,
        "underPct": round(100 - over_pct, 1),
        "lean": "over" if shift > 0.005 else "under" if shift < -0.005 else "neutral",
        "parkEdge": park_edge,
        "parkFactor": park["factor"] if park else None,
        "weatherAdj": round(weather_adj, 4) if weather_adj else None,
        "notes": notes,
        # These conditions have never been graded against totals results, so
        # they are shown as context and not as a priced edge.
        "unvalidated": True,
    }


def predict_total(game: dict[str, Any], lines: list[dict[str, Any]], enrichment: dict[str, Any]) -> dict[str, Any] | None:
    total_line = extract_total_line(lines)
    if total_line is None:
        return None

    league = _league_id(game)
    over_lean = 0.5
    detail_parts: list[str] = []

    pace = _scoring_pace_from_form(enrichment)
    if pace is not None:
        if pace >= total_line + 0.8:
            over_lean += 0.20
            detail_parts.append(f"Recent scoring pace ({pace:.1f}) runs hot vs the {total_line} line.")
        elif pace <= total_line - 0.8:
            over_lean -= 0.20
            detail_parts.append(f"Recent scoring pace ({pace:.1f}) runs cool vs the {total_line} line.")

    home_pitcher = game.get("homePitcher") or {}
    away_pitcher = game.get("awayPitcher") or {}
    if home_pitcher.get("era") is not None and away_pitcher.get("era") is not None:
        avg_era = (home_pitcher["era"] + away_pitcher["era"]) / 2
        if avg_era <= 3.6:
            over_lean -= 0.12
            detail_parts.append(f"Strong pitching matchup (avg ERA {avg_era:.2f}) favors the under.")
        elif avg_era >= 4.6:
            over_lean += 0.12
            detail_parts.append(f"Weaker pitching matchup (avg ERA {avg_era:.2f}) favors the over.")

    environment = run_environment(game, enrichment)
    if environment:
        over_lean += environment["shift"]
        detail_parts.extend(environment["notes"])

    home_adv = enrichment.get("homeAdvanced") or {}
    away_adv = enrichment.get("awayAdvanced") or {}
    if home_adv.get("runsPerGame") is not None and away_adv.get("runsPerGame") is not None:
        combined = home_adv["runsPerGame"] + away_adv["runsPerGame"]
        if combined >= total_line + 1.5:
            over_lean += 0.10
            detail_parts.append(f"Season scoring pace ({combined:.1f} combined R/G) leans over.")
        elif combined <= total_line - 1.5:
            over_lean -= 0.10
            detail_parts.append(f"Season scoring pace ({combined:.1f} combined R/G) leans under.")

    if league in {"nba", "wnba", "afl"}:
        home_overall = win_pct_from_record(game.get("homeRecord"))
        away_overall = win_pct_from_record(game.get("awayRecord"))
        if home_overall + away_overall > 1.05:
            over_lean += 0.08
            detail_parts.append("Both teams have strong records — higher-scoring game possible.")

    # Wider clamp range for more decisive predictions (30-70% instead of 35-65%)
    over_lean = clamp(over_lean, 0.30, 0.70)
    under_lean = 1.0 - over_lean
    pick = "Over" if over_lean >= under_lean else "Under"
    confidence = max(over_lean, under_lean) * 100
    pick_side = pick.lower()

    return {
        "line": total_line,
        "pick": f"{pick} {total_line}",
        "pickSide": pick_side,
        "overPct": round(over_lean * 100, 1),
        "underPct": round(under_lean * 100, 1),
        "confidence": round(confidence, 1),
        # None on an SBR-sourced line, which has nowhere to carry a price.
        # Present when ESPN core supplied it -- see extract_total_price.
        "odds": extract_total_price(lines, pick_side),
        "detail": " ".join(detail_parts) if detail_parts else f"Model leans {pick_side} vs market total {total_line}.",
    }


def _model_market_edge(
    *,
    predicted_side: str,
    home_prob: float,
    away_prob: float,
    market_home: float | None,
    market_away: float | None,
) -> dict[str, Any] | None:
    if market_home is None or market_away is None:
        return None
    model_side = home_prob if predicted_side == "home" else away_prob
    market_side = market_home if predicted_side == "home" else market_away
    edge = (model_side - market_side) * 100
    return {
        "modelPct": round(model_side * 100, 1),
        "marketPct": round(market_side * 100, 1),
        "edgePct": round(edge, 1),
        "edgeLabel": f"{edge:+.1f}% vs market",
        "favorsModel": abs(edge) >= 3,
    }


def _probability_components(
    *,
    resolved: dict[str, Any],
    heuristic_home: float,
    model_inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Explain where the published probability came from.

    Shows the market and the heuristic alongside the published number so a large
    gap between them is visible rather than buried.
    """
    components: list[dict[str, Any]] = [
        {
            "source": "Fitted model" if resolved["method"] == "fitted" else "Heuristic fallback",
            "detail": resolved["detail"],
            "homePct": round(resolved["binaryHome"] * 100, 1),
        }
    ]

    implied_home = model_inputs.get("impliedHome")
    if implied_home is not None:
        components.append(
            {
                "source": "Market (de-vigged)",
                "detail": "Consensus moneyline with the vig removed",
                "homePct": round(float(implied_home), 1),
            }
        )

    if resolved["method"] == "fitted":
        components.append(
            {
                "source": "Heuristic (reference only)",
                "detail": "Legacy hand-tuned logit, not used for the published number",
                "homePct": round(heuristic_home * 100, 1),
            }
        )

    return components


def _moneyline_quotes(lines: list[dict[str, Any]], side: str) -> list[int]:
    """Every usable book quote for one side, outliers already dropped."""
    key = {"home": ("home", "homeOdds"), "away": ("away", "awayOdds"), "draw": ("draw", "drawOdds")}.get(side)
    if not key:
        return []

    quotes: list[int] = []
    for line in lines or []:
        if "MoneyLine" not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        odds = _line_odds_value(current, *key)
        if odds is not None:
            quotes.append(int(odds))

    return _usable_quotes(quotes)


def _best_price_for_side(lines: list[dict[str, Any]], side: str) -> int | None:
    """Best available American price for one side, across every book quoted.

    Best means the largest payout, which is the highest decimal odds -- not the
    highest American number, since -110 beats -150 but +120 beats both.
    """
    quotes = _moneyline_quotes(lines, side)
    if not quotes:
        return None
    return max(quotes, key=lambda odds: american_to_decimal(odds))


def quote_spread(lines: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    """How much taking the best book is worth, against taking a typical one.

    The build already shops: `_best_price_for_side` returns the best quote
    across every book on the game. What it does not do is record what the
    alternatives were, so the value of shopping has never been measurable --
    the comparison exists for a few microseconds inside one build and is then
    discarded.

    That matters because the coverage is uneven. The Odds API emits every book
    it has, so those games are genuinely shopped; ESPN core reports a single
    book -- every pricing line in the logs reads "via DraftKings" -- and
    SportsBookReview is one board. On the single-book games there is nothing to
    shop, and no way to know what is being left behind without a record of what
    a multi-book game looks like.

    `gainPct` is the difference in implied probability between the best quote
    and the median one, in percentage points. That unit is chosen so it lands
    on the same scale as closing line value, which is also implied-probability
    points -- and that comparison is the point. CLV currently runs at a median
    of -0.4 points, so a shopping gain of even one point would more than cover
    the ground the model loses to the close, and it would do so mechanically,
    with no modelling risk at all.

    None when the game carries no usable quote. A single-book game returns a
    spread of zero rather than None, because "one book, nothing to gain" is a
    finding and an absent record is not.
    """
    quotes = _moneyline_quotes(lines, side)
    if not quotes:
        return None
    best = max(quotes, key=lambda odds: american_to_decimal(odds))
    ordered = sorted(quotes, key=lambda odds: american_to_decimal(odds))
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        # Two books have no middle quote; the worse of the pair is the honest
        # stand-in for "what you would have got without shopping".
        else ordered[middle - 1]
    )
    best_implied = american_odds_to_implied(best)
    median_implied = american_odds_to_implied(median)
    gain = None
    if best_implied is not None and median_implied is not None:
        gain = round((median_implied - best_implied) * 100, 3)
    return {
        "books": len(quotes),
        "best": best,
        "median": median,
        "worst": ordered[0],
        "gainPct": gain,
    }


# Books disagree by a point or two on a moneyline, not by tens of points. A
# quote that far from the rest is bad data -- a stale line, a mis-parsed field,
# or odds matched to the wrong game -- and "best price" will happily pick it,
# because a garbage price always looks like the best one.
#
# This is not hypothetical: one WNBA game carried a +575 quote beside a -153,
# and the +575 was selected, producing a published EV of +278.9% where the
# real figure was -7.2%.
MAX_QUOTE_DISAGREEMENT_PTS = 25.0


def _usable_quotes(quotes: list[int]) -> list[int]:
    """Drop quotes that disagree with the rest of the book beyond all reason.

    With three or more, the median is trustworthy and outliers are dropped
    against it. With exactly two there is no way to tell which one is wrong,
    so a wild disagreement discards both: an unpriced game is honest, a
    confidently wrong price is not.
    """
    if len(quotes) < 2:
        return quotes
    implied = sorted(american_odds_to_implied(odds) * 100 for odds in quotes)
    if len(quotes) == 2:
        return [] if implied[1] - implied[0] > MAX_QUOTE_DISAGREEMENT_PTS else quotes
    middle = implied[len(implied) // 2]
    return [
        odds for odds in quotes
        if abs(american_odds_to_implied(odds) * 100 - middle) <= MAX_QUOTE_DISAGREEMENT_PTS
    ]


def _home_field_logit(game: dict[str, Any]) -> float:
    return HOME_FIELD_LOGIT.get(_league_id(game), 0.25)


def _home_field_detail(game: dict[str, Any]) -> str:
    league = _league_id(game)
    venue = game.get("venueName") or "home"
    if league == "epl":
        return f"{game.get('homeTeam')} play at {venue}, where home sides often perform better."
    if league == "afl":
        return f"{game.get('homeTeam')} have home-ground advantage at {venue}."
    if league in {"nba", "wnba"}:
        return f"{game.get('homeTeam')} have home-court advantage at {venue}."
    if league == "nfl":
        return f"{game.get('homeTeam')} play at {venue}, where home teams historically win more often."
    return f"{game.get('homeTeam')} play at {venue}, where home teams historically win more often."


def _build_reasons(
    game: dict[str, Any],
    *,
    predicted_side: str,
    home_prob: float,
    away_prob: float,
    enrichment: dict[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    league = _league_id(game)
    winner = _team_by_side(game, predicted_side)
    loser_side = "away" if predicted_side == "home" else "home"
    loser = _team_by_side(game, loser_side)

    home_overall = win_pct_from_record(game.get("homeRecord"))
    away_overall = win_pct_from_record(game.get("awayRecord"))
    if _edge_label(home_overall, away_overall) == predicted_side:
        reasons.append(
            {
                "title": "Better season record",
                "detail": (
                    f"{game.get('homeTeam')} ({game.get('homeRecord')}, {format_win_pct(game.get('homeRecord'))}) "
                    f"have outplayed {game.get('awayTeam')} ({game.get('awayRecord')}, {format_win_pct(game.get('awayRecord'))}) "
                    f"across the full season."
                ),
                "impact": "high",
                "favors": predicted_side,
            }
        )

    home_split = win_pct_from_record(game.get("homeHomeRecord"), home_overall)
    away_split = win_pct_from_record(game.get("awayRoadRecord"), away_overall)
    if _edge_label(home_split, away_split) == predicted_side:
        reasons.append(
            {
                "title": "Favorable home/road split",
                "detail": (
                    f"{game.get('homeTeam')} are {game.get('homeHomeRecord') or '?'} at home while "
                    f"{game.get('awayTeam')} are {game.get('awayRoadRecord') or '?'} on the road."
                ),
                "impact": "medium",
                "favors": predicted_side,
            }
        )

    home_pitcher = game.get("homePitcher") or {}
    away_pitcher = game.get("awayPitcher") or {}
    home_era = home_pitcher.get("era")
    away_era = away_pitcher.get("era")
    if home_era is not None and away_era is not None:
        better_side = "home" if home_era < away_era else "away" if away_era < home_era else "even"
        if better_side == predicted_side:
            winner_pitcher = home_pitcher if predicted_side == "home" else away_pitcher
            loser_pitcher = away_pitcher if predicted_side == "home" else home_pitcher
            reasons.append(
                {
                    "title": "Starting pitching edge",
                    "detail": (
                        f"{winner} send out {winner_pitcher.get('name')} ({winner_pitcher.get('era'):.2f} ERA) "
                        f"against {loser_pitcher.get('name')} ({loser_pitcher.get('era'):.2f} ERA)."
                    ),
                    "impact": "high",
                    "favors": predicted_side,
                }
            )

    espn_home = enrichment.get("espnPredictorHome")
    espn_away = enrichment.get("espnPredictorAway")
    if espn_home is not None and espn_away is not None:
        espn_side = "home" if espn_home >= espn_away else "away"
        if espn_side == predicted_side:
            reasons.append(
                {
                    "title": "ESPN Matchup Predictor agrees",
                    "detail": (
                        f"ESPN's model gives {game.get('homeTeam')} a {espn_home:.1f}% chance and "
                        f"{game.get('awayTeam')} a {espn_away:.1f}% chance."
                    ),
                    "impact": "high",
                    "favors": predicted_side,
                    "source": "ESPN",
                }
            )

    home_form = enrichment.get("homeLastFive") or {}
    away_form = enrichment.get("awayLastFive") or {}
    home_form_pct = _last_five_pct(home_form.get("record"), league)
    away_form_pct = _last_five_pct(away_form.get("record"), league)
    if home_form_pct is not None and away_form_pct is not None:
        form_side = _edge_label(home_form_pct, away_form_pct)
        if form_side == predicted_side:
            winner_form = home_form if predicted_side == "home" else away_form
            loser_form = away_form if predicted_side == "home" else home_form
            streak = "-".join(winner_form.get("results") or [])
            reasons.append(
                {
                    "title": "Recent form trending up",
                    "detail": (
                        f"{winner} are {winner_form.get('record')} in their last five"
                        + (f" ({streak})" if streak else "")
                        + " while "
                        f"{loser} are {loser_form.get('record')}."
                    ),
                    "impact": "medium",
                    "favors": predicted_side,
                    "source": "ESPN",
                }
            )

    series = enrichment.get("seasonSeries") or {}
    if series.get("summary") and winner:
        summary = series.get("summary") or ""
        if winner.split()[-1].lower() in summary.lower() or winner.lower() in summary.lower():
            reasons.append(
                {
                    "title": "Head-to-head history",
                    "detail": f"In the regular-season series: {summary} ({series.get('seriesScore')}).",
                    "impact": "medium",
                    "favors": predicted_side,
                    "source": "ESPN",
                }
            )

    if predicted_side == "home":
        reasons.append(
            {
                "title": "Home-field advantage",
                "detail": _home_field_detail(game),
                "impact": "low",
                "favors": "home",
            }
        )

    home_key_inj = enrichment.get("homeKeyInjuries") or []
    away_key_inj = enrichment.get("awayKeyInjuries") or []
    home_major = enrichment.get("homeMajorInjuries") or []
    away_major = enrichment.get("awayMajorInjuries") or []

    if len(away_major) > len(home_major) and predicted_side == "home" and away_major:
        names = ", ".join(f"{item['player']} ({item['status']})" for item in away_major[:3])
        reasons.append(
            {
                "title": "Opponent injury issues",
                "detail": f"{game.get('awayTeam')} missing or limited players: {names}.",
                "impact": "medium",
                "favors": "home",
                "source": "ESPN",
            }
        )
    elif len(home_major) > len(away_major) and predicted_side == "away" and home_major:
        names = ", ".join(f"{item['player']} ({item['status']})" for item in home_major[:3])
        reasons.append(
            {
                "title": "Opponent injury issues",
                "detail": f"{game.get('homeTeam')} missing or limited players: {names}.",
                "impact": "medium",
                "favors": "away",
                "source": "ESPN",
            }
        )

    if len(away_key_inj) > len(home_key_inj) + 1 and predicted_side == "home":
        reasons.append(
            {
                "title": "Injury advantage",
                "detail": (
                    f"{game.get('awayTeam')} have more notable injuries ({', '.join(away_key_inj[:3])}) "
                    f"than {game.get('homeTeam')} ({', '.join(home_key_inj[:3]) or 'none listed'})."
                ),
                "impact": "medium",
                "favors": "home",
                "source": "ESPN",
            }
        )
    elif len(home_key_inj) > len(away_key_inj) + 1 and predicted_side == "away":
        reasons.append(
            {
                "title": "Injury advantage",
                "detail": (
                    f"{game.get('homeTeam')} have more notable injuries ({', '.join(home_key_inj[:3])}) "
                    f"than {game.get('awayTeam')} ({', '.join(away_key_inj[:3]) or 'none listed'})."
                ),
                "impact": "medium",
                "favors": "away",
                "source": "ESPN",
            }
        )

    if enrichment.get("weather"):
        reasons.append(
            {
                "title": "Game conditions",
                "detail": f"Forecast at {game.get('venueName') or 'the ballpark'}: {enrichment['weather']}.",
                "impact": "low",
                "favors": "even",
                "source": "ESPN",
            }
        )

    home_adv = enrichment.get("homeAdvanced") or {}
    away_adv = enrichment.get("awayAdvanced") or {}
    if home_adv.get("powerRating") is not None and away_adv.get("powerRating") is not None:
        power_side = _edge_label(home_adv["powerRating"], away_adv["powerRating"])
        if power_side == predicted_side:
            reasons.append(
                {
                    "title": "Power rating edge",
                    "detail": (
                        f"Composite rating favors {winner}: "
                        f"{home_adv['powerRating']:.3f} vs {away_adv['powerRating']:.3f} "
                        f"(ESPN + MLB.com + form)."
                    ),
                    "impact": "high",
                    "favors": predicted_side,
                    "source": "Multi-source",
                }
            )

    if league_config := get_league(_league_id(game)):
        if league_config.id == "mlb" and home_adv.get("runDifferential") is not None and away_adv.get("runDifferential") is not None:
            rd_side = _edge_label(home_adv["runDifferential"], away_adv["runDifferential"])
            if rd_side == predicted_side:
                reasons.append(
                    {
                        "title": "Run differential",
                        "detail": (
                            f"{game.get('homeTeam')} {_format_plus_minus(home_adv['runDifferential'])} vs "
                            f"{game.get('awayTeam')} {_format_plus_minus(away_adv['runDifferential'])} (MLB.com)."
                        ),
                        "impact": "medium",
                        "favors": predicted_side,
                        "source": "MLB.com",
                    }
                )

    rest = enrichment.get("restDays") or {}
    if rest.get("home") is not None and rest.get("away") is not None:
        if rest["home"] > rest["away"] and predicted_side == "home":
            reasons.append(
                {
                    "title": "Rest advantage",
                    "detail": f"{game.get('homeTeam')} have {rest['home']} days rest vs {rest['away']} for {game.get('awayTeam')}.",
                    "impact": "low",
                    "favors": "home",
                    "source": "Schedule",
                }
            )
        elif rest["away"] > rest["home"] and predicted_side == "away":
            reasons.append(
                {
                    "title": "Rest advantage",
                    "detail": f"{game.get('awayTeam')} have {rest['away']} days rest vs {rest['home']} for {game.get('homeTeam')}.",
                    "impact": "low",
                    "favors": "away",
                    "source": "Schedule",
                }
            )

    h2h = enrichment.get("headToHead") or {}
    if h2h.get("summary") and winner and predicted_side in {"home", "away"}:
        home_h2h = h2h.get("homeSeriesWinPct")
        away_h2h = h2h.get("awaySeriesWinPct")
        if home_h2h is not None and away_h2h is not None:
            h2h_side = _edge_label(home_h2h, away_h2h)
            if h2h_side == predicted_side:
                reasons.append(
                    {
                        "title": "Season series edge",
                        "detail": f"{h2h['summary']} ({h2h.get('seriesScore')}).",
                        "impact": "medium",
                        "favors": predicted_side,
                        "source": "ESPN",
                    }
                )

    if league_config and league_config.id == "epl" and home_adv.get("goalDifference") is not None and away_adv.get("goalDifference") is not None:
        gd_side = _edge_label(home_adv["goalDifference"], away_adv["goalDifference"])
        if gd_side == predicted_side:
            reasons.append(
                {
                    "title": "Goal difference edge",
                    "detail": (
                        f"{game.get('homeTeam')} GD {_format_plus_minus(home_adv['goalDifference'])} vs "
                        f"{game.get('awayTeam')} {_format_plus_minus(away_adv['goalDifference'])}."
                    ),
                    "impact": "medium",
                    "favors": predicted_side,
                    "source": "ESPN",
                }
            )

    impact_rank = {"high": 0, "medium": 1, "low": 2}
    reasons.sort(key=lambda reason: (0 if reason.get("favors") == predicted_side else 1, impact_rank.get(reason.get("impact", "low"), 9)))
    return reasons


# How each fitted feature reads in prose. Keys match model_fit's feature names.
DRIVER_LABELS = {
    "homeField": ("home-field advantage", "playing at home"),
    "strengthDiff": ("team strength", "the gap in team strength"),
    "marketLogit": ("the betting market", "where the market has priced this game"),
    "pitchingDiff": ("starting pitching", "the starting pitching matchup"),
    "restDiff": ("rest", "the rest advantage"),
    "injuryDiff": ("injuries", "the injury picture"),
    "injurySeverityDiff": ("injuries", "the weight of who is unavailable"),
    "b2bDiff": ("schedule", "the back-to-back schedule"),
}


def _why_from_drivers(
    drivers: list[dict[str, Any]] | None,
    predicted_winner: str | None,
    predicted_side: str,
) -> str | None:
    """Explain the pick using the terms the model actually weighed.

    The narrative used to be assembled from whatever enrichment was available --
    ESPN's predictor, last-five form, head-to-head -- none of which the fitted
    model reads. The explanation and the number had quietly come apart, so this
    builds the sentence from the real logit decomposition instead.
    """
    if not drivers or not predicted_winner or predicted_side not in {"home", "away"}:
        return None

    # A positive contribution favours home, so flip the sign for an away pick.
    sign = 1.0 if predicted_side == "home" else -1.0
    supporting = [
        item
        for item in drivers
        if item.get("available") and item.get("contribution") is not None
        and item["contribution"] * sign > 0.01
    ]
    if not supporting:
        return (
            f"{predicted_winner} are a marginal pick — no single factor separates "
            f"these teams by much."
        )

    supporting.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    phrases = [
        DRIVER_LABELS.get(item["feature"], (item["feature"], item["feature"]))[1]
        for item in supporting[:3]
    ]

    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = f"{', '.join(phrases[:-1])} and {phrases[-1]}"
    return f"{predicted_winner} are favoured on {joined}."


def _build_why_they_win(game: dict[str, Any], reasons: list[dict[str, Any]], predicted_winner: str | None) -> str:
    if not predicted_winner:
        return "Not enough data to explain this pick yet."

    top_reasons = [reason for reason in reasons if reason.get("favors") in {"home", "away"}][:4]
    if not top_reasons:
        # "match" for soccer, "game" for everything else.
        noun = "match" if _league_id(game) == "epl" else "game"
        return (
            f"{predicted_winner} are slightly favored in a close {noun} based on combined team strength "
            f"and home-field factors."
        )

    joined = "; ".join(reason["detail"] for reason in top_reasons[:3])
    return f"{predicted_winner} are projected to win because {joined}"


def predict_game(game: dict[str, Any]) -> dict[str, Any]:
    factors: list[dict[str, Any]] = []
    enrichment = game.get("enrichment") or {}
    league_config = get_league(_league_id(game))

    home_overall = win_pct_from_record(game.get("homeRecord"))
    away_overall = win_pct_from_record(game.get("awayRecord"))
    home_split = win_pct_from_record(game.get("homeHomeRecord"), home_overall)
    away_split = win_pct_from_record(game.get("awayRoadRecord"), away_overall)

    home_pitcher = game.get("homePitcher") or {}
    away_pitcher = game.get("awayPitcher") or {}
    home_era = home_pitcher.get("era")
    away_era = away_pitcher.get("era")

    logit = 0.0

    record_diff = home_overall - away_overall
    logit += record_diff * 3.2
    factors.append(
        {
            "label": "Season record",
            "detail": f"{game.get('homeTeam')} {game.get('homeRecord') or '?'} vs {game.get('awayTeam')} {game.get('awayRecord') or '?'}",
            "edge": _edge_label(home_overall, away_overall),
        }
    )

    split_diff = home_split - away_split
    logit += split_diff * 2.4
    factors.append(
        {
            "label": "Home/road splits",
            "detail": f"Home {game.get('homeHomeRecord') or '?'} vs away {game.get('awayRoadRecord') or '?'}",
            "edge": _edge_label(home_split, away_split),
        }
    )

    logit += _home_field_logit(game)
    factors.append(
        {
            "label": "Home-field advantage",
            "detail": "Historical home edge applied",
            "edge": "home",
        }
    )

    if league_config.supports_pitchers and home_era is not None and away_era is not None:
        pitching = enrichment.get("mlbPitching") or {}
        home_fip = pitching.get("homePitcherFip") or home_pitcher.get("fip")
        away_fip = pitching.get("awayPitcherFip") or away_pitcher.get("fip")
        home_recent = pitching.get("homePitcherRecentEra")
        away_recent = pitching.get("awayPitcherRecentEra")

        era_diff = away_era - home_era
        logit += era_diff * 0.38
        if home_fip is not None and away_fip is not None:
            logit += (away_fip - home_fip) * 0.22
        if home_recent is not None and away_recent is not None:
            logit += (away_recent - home_recent) * 0.25

        detail_parts = [
            f"{home_pitcher.get('name') or 'Home SP'} ({home_era:.2f} ERA",
            f"{away_pitcher.get('name') or 'Away SP'} ({away_era:.2f} ERA",
        ]
        if home_fip is not None and away_fip is not None:
            detail_parts[0] += f", {home_fip:.2f} FIP"
            detail_parts[1] += f", {away_fip:.2f} FIP"
        detail_parts[0] += ")"
        detail_parts[1] += ")"
        factors.append(
            {
                "label": "Starting pitching",
                "detail": f"{detail_parts[0]} vs {detail_parts[1]}",
                "edge": _edge_label(-home_era, -away_era),
            }
        )

    home_form = enrichment.get("homeLastFive") or {}
    away_form = enrichment.get("awayLastFive") or {}
    home_form_pct = _last_five_pct(home_form.get("record"), league_config.id)
    away_form_pct = _last_five_pct(away_form.get("record"), league_config.id)
    if home_form_pct is not None and away_form_pct is not None:
        logit += (home_form_pct - away_form_pct) * 1.8
        factors.append(
            {
                "label": "Last five games",
                "detail": f"{game.get('homeTeam')} {home_form.get('record')} vs {game.get('awayTeam')} {away_form.get('record')}",
                "edge": _edge_label(home_form_pct, away_form_pct),
            }
        )

    injury_adj = _injury_logit_adjustment(enrichment, league_config.id)
    if injury_adj:
        logit += injury_adj
        home_load = _weighted_injury_score(enrichment.get("homeMajorInjuries") or [], league_config.id)
        away_load = _weighted_injury_score(enrichment.get("awayMajorInjuries") or [], league_config.id)
        factors.append(
            {
                "label": "Injury impact",
                "detail": f"Weighted injury load: home {home_load:.1f} vs away {away_load:.1f}",
                "edge": "home" if injury_adj > 0 else "away" if injury_adj < 0 else "even",
            }
        )

    streak_adj = _streak_logit_adjustment(enrichment)
    if streak_adj:
        logit += streak_adj
        factors.append(
            {
                "label": "Win/loss streak",
                "detail": "MLB.com / standings streak momentum",
                "edge": "home" if streak_adj > 0 else "away" if streak_adj < 0 else "even",
            }
        )

    lineup_adj = _lineup_logit_adjustment(game, league_config.id, enrichment)
    if lineup_adj:
        logit += lineup_adj
        factors.append(
            {
                "label": "Lineup quality",
                "detail": "Confirmed lineup strength vs opponent",
                "edge": "home" if lineup_adj > 0 else "away" if lineup_adj < 0 else "even",
            }
        )

    weather_adj = _weather_win_logit_adjustment(game, enrichment, league_config.id)
    if weather_adj:
        logit += weather_adj
        factors.append(
            {
                "label": "Weather (outdoor)",
                "detail": (enrichment.get("weatherImpact") or {}).get("summary") or "Weather-adjusted edge",
                "edge": "home" if weather_adj > 0 else "away" if weather_adj < 0 else "even",
            }
        )

    schedule_adj = schedule_flags_logit_adjustment(enrichment)
    if schedule_adj:
        logit += schedule_adj
        home_flags = enrichment.get("homeScheduleFlags") or {}
        away_flags = enrichment.get("awayScheduleFlags") or {}
        factors.append(
            {
                "label": "Schedule fatigue",
                "detail": (
                    f"Home B2B={home_flags.get('backToBack')} · Away B2B={away_flags.get('backToBack')}"
                ),
                "edge": "home" if schedule_adj > 0 else "away" if schedule_adj < 0 else "even",
            }
        )

    league_adj = league_metrics_logit_adjustment(enrichment, league_config.id)
    if league_adj:
        logit += league_adj
        metrics = enrichment.get("leagueMetrics") or {}
        factors.append(
            {
                "label": "League advanced metrics",
                "detail": str(metrics) if metrics else "Pace/efficiency/xG proxies",
                "edge": "home" if league_adj > 0 else "away" if league_adj < 0 else "even",
            }
        )

    if league_config.id == "mlb":
        mlb_pitch_adj = mlb_pitching_logit_adjustment(game, enrichment)
        if mlb_pitch_adj:
            logit += mlb_pitch_adj
            factors.append(
                {
                    "label": "Pitcher/bullpen depth",
                    "detail": "MLB Stats API SP and bullpen ERA context",
                    "edge": "home" if mlb_pitch_adj > 0 else "away" if mlb_pitch_adj < 0 else "even",
                }
            )

    advanced_adj = _advanced_logit_adjustment(enrichment, league_config.id)
    if advanced_adj:
        logit += advanced_adj
        home_adv = enrichment.get("homeAdvanced") or {}
        away_adv = enrichment.get("awayAdvanced") or {}
        home_power = home_adv.get("powerRating")
        away_power = away_adv.get("powerRating")
        if home_power is not None and away_power is not None:
            power_detail = f"Power {home_power:.3f} vs {away_power:.3f} (ESPN/MLB.com)"
        else:
            power_detail = "Multi-source team analytics"
        factors.append(
            {
                "label": "Advanced team profile",
                "detail": power_detail,
                "edge": "home" if advanced_adj > 0 else "away" if advanced_adj < 0 else "even",
            }
        )

    rest_adj = _rest_logit_adjustment(enrichment)
    if rest_adj:
        logit += rest_adj
        rest = enrichment.get("restDays") or {}
        factors.append(
            {
                "label": "Rest days",
                "detail": f"Home {rest.get('home', '?')} vs away {rest.get('away', '?')} days rest",
                "edge": "home" if rest_adj > 0 else "away" if rest_adj < 0 else "even",
            }
        )

    h2h_adj = _head_to_head_logit_adjustment(enrichment)
    if h2h_adj:
        logit += h2h_adj
        h2h = enrichment.get("headToHead") or {}
        factors.append(
            {
                "label": "Season series",
                "detail": h2h.get("summary") or "Head-to-head history",
                "edge": "home" if h2h_adj > 0 else "away" if h2h_adj < 0 else "even",
            }
        )

    espn_home = enrichment.get("espnPredictorHome")
    espn_away = enrichment.get("espnPredictorAway")
    if espn_home is not None and espn_away is not None:
        factors.append(
            {
                "label": "ESPN predictor",
                "detail": f"{espn_home:.1f}% home / {espn_away:.1f}% away",
                "edge": _edge_label(espn_home / 100.0, espn_away / 100.0),
            }
        )

    # The accumulated `logit` above still drives the factor edge labels shown in
    # the UI, but it is only used as the probability when no fitted weights are
    # available -- it stacks correlated features and is badly overconfident.
    heuristic_home = sigmoid(logit)
    model_inputs = extract_model_inputs(game)
    resolved = resolve_probabilities(
        game=game,
        model_inputs=model_inputs,
        heuristic_home=heuristic_home,
        enrichment=enrichment,
        league_config=league_config,
        league=league_config.id,
        legacy_calibrate=calibrate_probability,
    )

    true_probs = {
        "home": resolved["home"],
        "away": resolved["away"],
        "draw": resolved["draw"],
        "homePct": round(resolved["home"] * 100, 1),
        "awayPct": round(resolved["away"] * 100, 1),
        "drawPct": round(resolved["draw"] * 100, 1) if resolved["draw"] is not None else None,
        "method": resolved["method"],
        "components": _probability_components(
            resolved=resolved,
            heuristic_home=heuristic_home,
            model_inputs=model_inputs,
        ),
    }

    factors.append(
        {
            "label": "Model probability",
            "detail": (
                f"{resolved['detail']}: {true_probs['homePct']}% home / {true_probs['awayPct']}% away"
                + (f" / {true_probs['drawPct']}% draw" if true_probs.get("drawPct") is not None else "")
            ),
            "edge": _edge_label(true_probs["home"], true_probs["away"]),
        }
    )

    # resolve_probabilities already applied calibration (fallback path) or
    # returned a fitted estimate that must not be shrunk again, and it split the
    # three-way soccer outcome without pulling the draw toward 50%. The numbers
    # are final here; they only need to sum to one.
    home_prob = resolved["home"]
    away_prob = resolved["away"]
    draw_prob = resolved["draw"] or 0.0 if league_config.supports_draw else 0.0

    prob_total = home_prob + away_prob + draw_prob
    if prob_total > 0:
        home_prob /= prob_total
        away_prob /= prob_total
        draw_prob /= prob_total

    # Pre-calibration probabilities, logged so the calibrator can be fitted on
    # raw output rather than on its own previous corrections. rawHomeWinPct is
    # the one calibration actually fits on -- home-probability space, so the
    # curve is valid on both sides of 50%.
    raw_home_pct = round(resolved["rawBinaryHome"] * 100, 1)
    raw_confidence = round(max(resolved["rawBinaryHome"], 1.0 - resolved["rawBinaryHome"]) * 100, 1)

    outcomes = [
        ("home", home_prob, game.get("homeTeam")),
        ("away", away_prob, game.get("awayTeam")),
    ]
    if league_config.supports_draw and draw_prob:
        outcomes.append(("draw", draw_prob, "Draw"))

    predicted_side, best_prob, predicted_winner = max(outcomes, key=lambda item: item[1])
    confidence = best_prob * 100
    home_pct = round(home_prob * 100, 1)
    away_pct = round(away_prob * 100, 1)

    reasons = _build_reasons(
        game,
        predicted_side=predicted_side if predicted_side in {"home", "away"} else "home",
        home_prob=home_prob,
        away_prob=away_prob,
        enrichment=enrichment,
    )
    if predicted_side == "draw":
        reasons.insert(
            0,
            {
                "title": "Draw is the top outcome",
                "detail": f"Model estimates a {round(draw_prob * 100, 1)}% chance of a draw.",
                "impact": "high",
                "favors": "even",
                "source": "Model",
            },
        )
    # Prefer the model's own decomposition. _build_why_they_win reads whichever
    # enrichment happened to be present, which the fitted model does not use.
    if predicted_side == "draw":
        why_they_win = (
            f"Draw is the most likely result ({round(draw_prob * 100, 1)}%) based on form and matchup data."
        )
    else:
        why_they_win = _why_from_drivers(
            resolved.get("drivers"), predicted_winner, predicted_side
        ) or _build_why_they_win(game, reasons, predicted_winner)

    # Everything gathered by the enrichment pipeline is still worth showing, but
    # it is context around the pick rather than the cause of it. Flagging it
    # stops the panel from implying the model weighed factors it never read.
    driven_by_model = bool(resolved.get("drivers"))
    for reason in reasons:
        reason["usedInPick"] = not driven_by_model

    data_sources = ["ESPN scoreboard"]
    if enrichment:
        data_sources.extend(enrichment.get("sources") or [])
    data_sources.append("Probability model")

    pick_pct = {
        "homePct": home_pct,
        "awayPct": away_pct,
        "drawPct": round(draw_prob * 100, 1) if draw_prob else None,
        "method": "Model data only (records, form, injuries, advanced stats)",
    }
    implied_probs = compute_implied_probabilities(game.get("lines") or [])
    probabilities: dict[str, Any] = {
        "true": true_probs,
        "pick": pick_pct,
        "implied": implied_probs if implied_probs.get("available") else {"available": False},
    }
    team_probabilities = _build_team_probabilities(
        true_probs=true_probs,
        implied_probs=probabilities["implied"],
        blended=pick_pct,
    )

    # Expected value on the picked side at the best price on offer. Percentage
    # points of edge do not tell you whether to bet: the same 3-point edge is
    # worth +4.0% per unit at -300 and +10.5% at +250. This does.
    best_odds = _best_price_for_side(game.get("lines") or [], predicted_side)
    value = assess_price(
        best_prob, best_odds, kelly_probability=kelly_band_probability(confidence)
    )

    result: dict[str, Any] = {
        "predictedWinner": predicted_winner,
        "predictedSide": predicted_side,
        "homeWinPct": home_pct,
        "awayWinPct": away_pct,
        "confidence": round(confidence, 1),
        "rawConfidence": raw_confidence,
        "rawHomeWinPct": raw_home_pct,
        "confidenceLabel": confidence_label(confidence),
        "probabilityMethod": resolved["method"],
        # None when the game is unpriced -- there is no EV without a price.
        "value": value,
        # Per-feature logit contributions behind this number, so the displayed
        # explanation and the probability cannot drift apart.
        "drivers": resolved.get("drivers"),
        "outcomeLabel": f"{predicted_winner} to win" if predicted_side != "draw" else "Draw",
        "whyTheyWin": why_they_win,
        "reasons": reasons,
        "factors": factors,
        "dataSources": sorted(set(data_sources)),
        "probabilities": probabilities,
        "teamProbabilities": team_probabilities,
    }
    if league_config.supports_draw and draw_prob:
        result["drawWinPct"] = round(draw_prob * 100, 1)
    result["features"] = extract_prediction_features(game, result)
    return result


# Standard deviation of a game's final margin, in points. Converting a win
# probability to a spread is spread = -InverseNormalCDF(p) * sigma, which is
# the standard approach and, unlike a linear points-per-percent factor, stays
# sane at the extremes: a linear map turned an 80% home side into a 21-point
# favourite where the correct answer is about 9.
#
# Leagues absent from this map get no spread pick: baseball and hockey run a
# fixed 1.5 line where this mapping does not apply, and soccer uses handicaps
# that need their own treatment.
# Standard deviation of final margin, used to turn a win probability into a
# points line. Measured from graded results in docs/data/accuracy.json where
# there are enough of them, rather than assumed:
#
#     league   n     mean margin   SD
#     mlb    503        +0.04     4.83
#     wnba    97        +2.00    13.49   (was 10.0 -- understated)
#     afl     46        +8.28    40.05   (was 36.0 -- understated)
#
# CAREFUL: the table above is the WRONG STATISTIC, and the NFL and NBA entries
# below -- which came from published figures -- are the right one.
#
# Both uses of this number need the spread of a single game's margin around
# THAT GAME's expected margin, a residual. Converting a probability to a line
# solves p = Phi(mu / sigma) for mu, and the cover check is
# Phi((line - model_spread) / sigma); in each case the mean is the matchup's own
# expected margin, not the league's. The SD taken across all games is larger,
# because it also carries the game-to-game variation in team strength:
#
#     Var(margin over all games) = Var(expected margin) + Var(residual)
#
# Measured over completed 2025-26 seasons with scripts/measure_margin_sd.py,
# the across-all-games figures are nba 16.21 (n=1059), nfl 14.18 (n=272, exactly
# a full regular season, which is the check that the extraction is right) and
# wnba 15.16 (n=288). Substituting nba 16.21 here makes an 80% favourite a
# 13.6-point favourite, where the market prices that matchup around 9 or 10 --
# tests/test_model_fixes.py::test_line_stays_realistic_at_the_extremes fails on
# exactly that, and it is right to. NBA's 11.5 is the residual and is correct.
#
# OPEN, and a real error the other way: wnba 13.49 and afl 40.05 were measured
# from graded games as an across-all-games SD -- the same wrong statistic -- so
# both are too large. That is a conservative error: cover probabilities are
# pulled toward 50% and lines come out too steep, so it costs picks rather than
# making bad ones. Do not "fix" these by measuring raw margins harder.
#
# scripts/estimate_margin_residual.py does the conversion without needing
# closing spreads, using Var(margin) = Var(expected margin) + Var(residual)
# and reading Var(expected margin) off the spread of market win probabilities
# already in the graded log. It reproduces NBA's 11.5 from NBA's raw 16.21,
# which is the check that it works.
#
# It cannot run yet: it needs ~100 priced graded games per league and WNBA has
# 23, AFL none, because AFL only gained a price source recently. Re-run it once
# the log fills and promote what it reports.
#
# MLB and EPL are deliberately absent. Their handicap is a FIXED line -- the
# baseball runline is always +/-1.5 -- so mapping a win probability through a
# normal curve is the wrong tool, and the data says so plainly: run margins are
# discrete, cannot be zero, and pile up at 1, with 29% of 503 graded games
# decided by exactly one run. A normal curve cannot represent that spike.
# Baseball is handled by `predict_runline` below instead.
MARGIN_STD_DEV = {
    "nfl": 13.5,
    "nba": 11.5,
    "wnba": 13.49,
    "afl": 40.05,
}

# P(winner covers -1.5 | they won), measured on 503 graded MLB games: 147 of the
# 503 decided games were won by exactly one run, so 70.8% of wins clear the
# runline. This is what makes a runline model possible without a continuous
# margin distribution -- the line never moves, so only this one number is needed.
#
# It is applied symmetrically even though the split is not quite even (home
# winners take 33.9% of their wins by one run, away winners 24.6%). That gap is
# roughly the home-field edge showing up in close games and at n=251/252 it is
# not separated from noise, so splitting it would be fitting the sample.
RUNLINE_COVER_GIVEN_WIN = 0.7078
RUNLINE = 1.5


def predict_runline(
    game: dict[str, Any],
    lines: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    """Baseball's fixed +/-1.5 handicap, from the win probability.

    A separate function from `predict_spread` because the runline is not a
    spread. It never moves, so there is nothing to solve for; the question is
    only whether the favourite wins by two or more. That makes it a single
    conditional probability rather than a normal-curve inversion, and it is
    measured (`RUNLINE_COVER_GIVEN_WIN`) rather than assumed.
    """
    if _league_id(game) != "mlb":
        return None
    if extract_spread_line(lines) is None:
        return None

    true_probs = (prediction.get("probabilities") or {}).get("true") or {}
    home_prob = true_probs.get("home")
    away_prob = true_probs.get("away")
    if home_prob is None or away_prob is None:
        return None
    two_way = home_prob + away_prob
    if two_way <= 0:
        return None

    p_home = home_prob / two_way
    # Laying -1.5 needs a win by two or more; taking +1.5 also cashes when the
    # underdog loses by exactly one, which is why the favourite's runline price
    # is so much shorter than its moneyline.
    home_covers = p_home * RUNLINE_COVER_GIVEN_WIN
    away_covers = 1.0 - home_covers

    favourite_home = p_home >= 0.5
    pick_side = "home" if home_covers >= away_covers else "away"

    # The favourite lays the runs, the underdog takes them -- so the number
    # attached to a pick depends on whether THAT side is favoured, not on which
    # side was picked. Deriving it from pick_side alone printed "Padres -1.5"
    # for a side the model had at 40%, which is the opposite bet.
    picked_side_is_favourite = (pick_side == "home") == favourite_home
    line_for_pick = -RUNLINE if picked_side_is_favourite else RUNLINE

    team = game.get("homeTeam") if pick_side == "home" else game.get("awayTeam")
    return {
        "line": -RUNLINE if favourite_home else RUNLINE,
        "pick": f"{team} {line_for_pick:+.1f}",
        "pickSide": pick_side,
        "homePct": round(home_covers * 100, 1),
        "awayPct": round(away_covers * 100, 1),
        "market": "runline",
        # The cover probability for the side actually taken. It was already
        # computed here and quoted in the detail text below; leaving the field
        # None meant nothing downstream could price the pick, so the runline
        # could never be compared against the moneyline. Still `unvalidated`:
        # it is derived from the measured 70.8% clear rate, not calibrated
        # against its own graded record the way the moneyline is.
        "confidence": round(max(home_covers, away_covers) * 100, 1),
        "unvalidated": True,
        # Baseball runlines go through this function rather than predict_spread,
        # so without this call there is no route to a price at all.
        #
        # With it there is still no price, and that is a source problem rather
        # than a bug here. Measured 2026-08-05: ESPN core returns the runline
        # as {"home": "-1.5", "away": "+1.5"} with no juice attached, where the
        # same endpoint gives WNBA {"home": "-6.5 (-112)"}. So this reads None
        # on every MLB game and the runline still cannot be valued against the
        # moneyline. Pricing it needs a source that publishes the number --
        # see the note above the side-market pass in mlb_data.py.
        "odds": extract_spread_price(lines, pick_side),
        "detail": (
            f"Runline {line_for_pick:+.1f}. Model gives {team} "
            f"{max(home_covers, away_covers) * 100:.0f}% to cover, from a "
            f"{p_home * 100:.0f}% home win probability and the measured 70.8% of "
            f"wins that clear 1.5 runs."
        ),
    }


def predict_spread(
    game: dict[str, Any],
    lines: list[dict[str, Any]],
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert the model's win probability to a point spread and compare to market.

    Takes the prediction, not the enrichment. It previously read
    ``enrichment["probabilities"]`` -- a key that only ever exists on the
    prediction -- so both sides defaulted to 0.5. The model line was therefore
    always exactly 0.0, the "edge" was always minus the market line, and
    confidence was a function of the spread's size rather than of anything the
    model knew.
    """
    league = _league_id(game)
    margin_sd = MARGIN_STD_DEV.get(league)
    if margin_sd is None:
        return None

    spread_line = extract_spread_line(lines)
    if spread_line is None:
        return None

    true_probs = (prediction.get("probabilities") or {}).get("true") or {}
    home_prob = true_probs.get("home")
    away_prob = true_probs.get("away")
    if home_prob is None or away_prob is None:
        return None

    # Renormalise across the two sides in case a draw probability was carved out,
    # then map through the inverse normal CDF. A home favourite carries a
    # negative line, hence the sign flip.
    two_way = home_prob + away_prob
    if two_way <= 0:
        return None
    p_home = min(0.99, max(0.01, home_prob / two_way))
    model_spread = -NormalDist().inv_cdf(p_home) * margin_sd
    edge = model_spread - spread_line

    if abs(edge) < 0.5:
        pick_side, pick_text = "push", "No lean"
    elif edge < 0:
        # Model makes the home side stronger than the market does.
        pick_side, pick_text = "home", f"Home {spread_line:+.1f}"
    else:
        # The away side takes the opposite number to the home line.
        pick_side, pick_text = "away", f"Away {-spread_line:+.1f}"

    # Probability the picked side covers, from the same normal margin model that
    # produced modelLine -- not a second, inconsistent estimate.
    #
    # modelLine is -inv_cdf(p_home) * sigma, so the model's expected home margin
    # is -modelLine. The home side covers a home line L when margin + L > 0, so
    #   P(home covers) = P(margin > -L) = Phi((L - modelLine) / sigma).
    # The away side takes the complement, since exactly one of them covers on a
    # half-point line.
    cover_home = NormalDist().cdf((spread_line - model_spread) / margin_sd)
    cover = cover_home if pick_side == "home" else 1.0 - cover_home
    confidence = round(cover * 100, 1) if pick_side in ("home", "away") else None

    return {
        "line": spread_line,
        "modelLine": round(model_spread, 1),
        "pick": pick_text,
        "pickSide": pick_side,
        "edgePoints": round(edge, 1),
        # Derived from the margin model above rather than left blank. It is a
        # real probability, but it inherits that model's assumptions -- a normal
        # margin with a fixed per-league sigma -- and unlike the moneyline it has
        # never been calibrated against its own graded record. `unvalidated`
        # stays true so nothing downstream can mistake it for a fitted number.
        "confidence": confidence,
        "unvalidated": True,
        # None on a push (no side taken) or an SBR-sourced line, which has
        # nowhere to carry a price. Present when ESPN core supplied it.
        "odds": extract_spread_price(lines, pick_side) if pick_side in ("home", "away") else None,
        "detail": (
            f"Model line {model_spread:+.1f} vs market {spread_line:+.1f} "
            f"({edge:+.1f} pts)."
        ),
    }



def _price_side_market(block: dict[str, Any] | None) -> None:
    """Attach the same value block the moneyline carries, in place.

    Percentage points of edge are not comparable across markets any more than
    across prices: a 5-point edge on a -250 favourite and a 5-point edge on an
    over at -108 are worth very different amounts. Running every market through
    the same assess_price is what makes "which of these three is the best bet"
    a question with an answer.
    """
    if not block:
        return
    confidence = block.get("confidence")
    odds = block.get("odds")
    if confidence is None or odds is None:
        return
    value = assess_price(float(confidence) / 100.0, odds)
    if value:
        block["value"] = value


def _market_options(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    """Every market on this game that carries a real, priced expected value."""
    options: list[dict[str, Any]] = []
    candidates = (
        ("moneyline", prediction.get("value"), prediction.get("outcomeLabel"), prediction.get("confidence")),
        ("total", (prediction.get("total") or {}).get("value"),
         (prediction.get("total") or {}).get("pick"), (prediction.get("total") or {}).get("confidence")),
        ("spread", (prediction.get("spread") or {}).get("value"),
         (prediction.get("spread") or {}).get("pick"), (prediction.get("spread") or {}).get("confidence")),
    )
    for market, value, label, confidence in candidates:
        if not value or value.get("evPct") is None:
            continue
        options.append(
            {
                "market": market,
                "pick": label,
                "confidence": confidence,
                "odds": value.get("odds"),
                "evPct": value.get("evPct"),
                "kellyPct": value.get("kellyPct"),
                "breakEvenPct": value.get("breakEvenPct"),
                "validated": market_is_validated(market),
                "gradedPriced": None if market == "moneyline" else market_priced_history(market),
                # The record behind this market, with its error bar, so the
                # card can say how thin the evidence is rather than presenting
                # a hit rate on 70-odd picks as a settled fact.
                # pricedPct and pricedStdErrPct are the pair the gate now reads,
                # so the card shows the same numbers the decision was made on
                # rather than the flattering blended ones beside them.
                "record": None if market == "moneyline" else {
                    key: record.get(key)
                    for key in ("pct", "stdErrPct", "pricedPct", "pricedStdErrPct",
                                "breakEvenPct", "beatsBreakEven",
                                "decided", "priced", "pricedRoiPct")
                } if (record := market_record(market)) else None,
            }
        )
    return options


def select_best_bet(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Rank this game's markets by expected value and name one to back.

    Gated rather than a bare argmax. The moneyline is fitted and calibrated
    against every graded game; the side markets are heuristics that have only
    recently started carrying prices at all. Letting an unvalidated market win
    the headline purely on a bigger EV number would present a hand-tuned lean
    as the model's best work.

    So the headline goes to the highest-EV market that has earned it, while
    every priced market -- validated or not -- is still ranked and returned, so
    a bigger unvalidated edge is visible rather than hidden. When nothing has
    a price there is no best bet, which is a fact and not a gap.
    """
    options = _market_options(prediction)
    if not options:
        return None
    options.sort(key=lambda option: option["evPct"], reverse=True)

    eligible = [
        option for option in options
        if option["validated"] and option["evPct"] > 0
    ]
    headline = eligible[0] if eligible else None
    top = options[0]

    return {
        "options": options,
        # None when no market both clears zero EV and is trusted enough to back.
        "pick": headline,
        # Set when the genuinely highest-EV market was held back by the gate,
        # so the card can say why rather than silently dropping it.
        "heldBack": top if headline is not None and top is not headline else None,
    }


def _enrich_missing(games: list[dict[str, Any]]) -> None:
    """Enrich only games that have not been enriched already, batched by league.

    This used to run per game inside the prediction loop, which broke two
    things. enrich_games_with_providers derives rest days and back-to-back
    flags from the *other* games it is given, so calling it with a single game
    left every team with no prior fixture -- rest came back None and
    backToBack False on every game, overwriting the correct values
    mlb_data.py had already computed from a 7-day window. It also refetched
    standings and team stats once per game instead of once per slate.
    """
    by_league: dict[str, list[dict[str, Any]]] = {}
    for game in games:
        # mlb_data.py enriches the whole slate with a schedule context before
        # predictions run; `sources` marks that work as done.
        if (game.get("enrichment") or {}).get("sources"):
            continue
        by_league.setdefault(game.get("league") or "mlb", []).append(game)

    for league, subset in by_league.items():
        try:
            enrich_games_with_providers(subset, league=league)
        except Exception:
            # Enrichment is best-effort; predictions degrade rather than fail.
            continue


def apply_predictions(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _enrich_missing(games)

    for game in games:
        prediction = predict_game(game)
        prediction["publishable"] = is_publishable_pick(prediction, game.get("league"))

        lines = game.get("lines", [])
        enrichment = game.get("enrichment", {})

        # Outside the `if lines` block on purpose: the ballpark and the forecast
        # do not need a market to be known, so this shows on every baseball card
        # rather than only the ones a book has priced.
        environment = run_environment(game, enrichment)
        if environment:
            prediction["runEnvironment"] = environment

        if lines:
            total_pred = predict_total(game, lines, enrichment)
            if total_pred:
                prediction["total"] = total_pred

            # Baseball's handicap is a fixed runline, not a spread, so it takes
            # its own model. Either way it lands on `prediction["spread"]`, which
            # is what the card renders.
            spread_pred = predict_runline(game, lines, prediction) or predict_spread(
                game, lines, prediction
            )
            if spread_pred:
                prediction["spread"] = spread_pred

        # Price every market the same way before comparing them, then name the
        # one worth backing. Without this the card showed three markets and an
        # edge for only one of them, so "which should I actually bet" had no
        # answer on the page.
        _price_side_market(prediction.get("total"))
        _price_side_market(prediction.get("spread"))
        best = select_best_bet(prediction)
        if best:
            prediction["bestBet"] = best

        game["prediction"] = prediction

    publishable = [
        game
        for game in games
        if is_publishable_pick(game.get("prediction"), game.get("league"))
    ]
    publishable.sort(key=lambda game: game.get("prediction", {}).get("confidence", 0), reverse=True)
    for index, game in enumerate(publishable, start=1):
        game["predictionRank"] = index

    # Track identity, not equality: `game in publishable` compared whole game
    # dicts field by field, which is both quadratic and wrong for two games that
    # happen to serialise identically.
    published_ids = {id(game) for game in publishable}
    for game in games:
        if id(game) in published_ids:
            continue
        game.pop("predictionRank", None)
        prediction = game.get("prediction") or {}
        prediction["publishable"] = False
        game["prediction"] = prediction

    return games
