"""Win probability model and human-readable reasoning for scheduled games."""

from __future__ import annotations

import math
import re
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
from data_providers.schedule_advanced import schedule_flags_logit_adjustment
from data_providers.enrich import enrich_games_with_providers
from shared_utils import parse_record, win_pct_from_record, format_record, format_win_pct

HOME_FIELD_LOGIT = {
    "mlb": 0.28,
    "nfl": 0.32,
    "nba": 0.24,
    "wnba": 0.24,
    "worldcup": 0.35,
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
    "worldcup": 0.10,
    "afl": 0.06,
}
DEFAULT_MARKET_BLEND_WEIGHT = 0.10

_CALIBRATION_PARAMS: dict[str, Any] | None = None


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
            "home": raw_home / raw_total,
            "away": raw_away / raw_total,
            "draw": (raw_draw / raw_total) if raw_draw else None,
        },
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


def _last_five_pct(record: str | None) -> float | None:
    if not record:
        return None
    return win_pct_from_record(record, default=-1.0) if parse_record(record) else None


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
        # Spread is typically in the "home" field (negative = home favorite)
        value = current.get("home") or current.get("away")
        if not value:
            continue
        text = str(value).replace("+", "").replace("−", "-").replace("–", "-")
        try:
            return float(text.split()[0])
        except ValueError:
            continue
    return None


def _get_calibration_params() -> dict[str, Any]:
    global _CALIBRATION_PARAMS
    if _CALIBRATION_PARAMS is None:
        _CALIBRATION_PARAMS = load_calibration_params()
    return _CALIBRATION_PARAMS


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


def _lineup_quality_score(
    game: dict[str, Any],
    side: str,
    league: str,
    enrichment: dict[str, Any],
) -> float | None:
    lineup = game.get(f"{side}Lineup") or {}
    batters = lineup.get("batters") or []
    if not batters:
        return None

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

    if averages and league == "mlb":
        avg_value = sum(averages) / len(averages)
        return avg_value * 1.45

    advanced = enrichment.get("homeAdvanced" if side == "home" else "awayAdvanced") or {}
    if league == "mlb":
        ops = advanced.get("opsProxy")
        if ops is not None:
            confirm_ratio = len(confirmed) / max(1, len(batters))
            return float(ops) * (0.75 + 0.25 * confirm_ratio)

    if league in {"nba", "wnba", "nfl", "afl"}:
        scoring = advanced.get("pointsPerGame")
        if scoring is not None:
            return float(scoring) / 100.0

    return len(confirmed) / 9.0


def _lineup_logit_adjustment(game: dict[str, Any], league: str, enrichment: dict[str, Any]) -> float:
    home_score = _lineup_quality_score(game, "home", league, enrichment)
    away_score = _lineup_quality_score(game, "away", league, enrichment)
    if home_score is None and away_score is None:
        return 0.0
    home_value = home_score if home_score is not None else 0.5
    away_value = away_score if away_score is not None else 0.5
    multiplier = 2.5 if league == "mlb" else 1.5
    return max(-0.35, min(0.35, (home_value - away_value) * multiplier))


def _weather_win_logit_adjustment(game: dict[str, Any], enrichment: dict[str, Any], league: str) -> float:
    if league != "mlb":
        return 0.0
    venue = (game.get("venueName") or "").lower()
    if any(token in venue for token in ("dome", "roof", "tropicana", "minute maid")):
        return 0.0
    run_env = (enrichment.get("weatherImpact") or {}).get("runEnvironmentAdj") or 0.0
    return max(-0.12, min(0.12, run_env * 2.0))


def extract_prediction_features(game: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    enrichment = game.get("enrichment") or {}
    league = _league_id(game)
    home_adv = enrichment.get("homeAdvanced") or {}
    away_adv = enrichment.get("awayAdvanced") or {}
    implied = compute_implied_probabilities(game.get("lines") or [])
    consensus = implied.get("consensus") or {} if implied.get("available") else {}
    true_probs = (prediction.get("probabilities") or {}).get("true") or {}
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
        "recordDiff": round(
            win_pct_from_record(game.get("homeRecord")) - win_pct_from_record(game.get("awayRecord")),
            4,
        ),
        "splitDiff": round(
            win_pct_from_record(game.get("homeHomeRecord"), 0.5)
            - win_pct_from_record(game.get("awayRoadRecord"), 0.5),
            4,
        ),
        "homePower": home_adv.get("powerRating"),
        "awayPower": away_adv.get("powerRating"),
        "homeInjuryLoad": round(_weighted_injury_score(enrichment.get("homeMajorInjuries") or [], league), 2),
        "awayInjuryLoad": round(_weighted_injury_score(enrichment.get("awayMajorInjuries") or [], league), 2),
        "homeRest": rest_days.get("home"),
        "awayRest": rest_days.get("away"),
        "homeBackToBack": home_flags.get("backToBack"),
        "awayBackToBack": away_flags.get("backToBack"),
        "impliedHome": consensus.get("homePct"),
        "impliedAway": consensus.get("awayPct"),
        "trueHome": true_probs.get("homePct"),
        "trueAway": true_probs.get("awayPct"),
        "confidence": prediction.get("confidence"),
        "predictedSide": prediction.get("predictedSide"),
        "hasLineup": data_coverage["lineup"],
        "dataCoverage": data_coverage,
        "mlbPitching": enrichment.get("mlbPitching"),
        "leagueMetrics": enrichment.get("leagueMetrics"),
    }


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
    scores: list[float] = []
    for side in ("homeLastFive", "awayLastFive"):
        for game in (enrichment.get(side) or {}).get("games") or []:
            score_text = game.get("score") or ""
            parts = str(score_text).replace("-", " ").split()
            for part in parts:
                try:
                    scores.append(float(part))
                except ValueError:
                    continue
    if not scores:
        return None
    return sum(scores) / len(scores)


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

    weather_impact = enrichment.get("weatherImpact") or {}
    run_env = weather_impact.get("runEnvironmentAdj")
    if run_env:
        over_lean += run_env
        if weather_impact.get("summary"):
            detail_parts.append(f"Weather: {weather_impact['summary']}.")

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

    return {
        "line": total_line,
        "pick": f"{pick} {total_line}",
        "pickSide": pick.lower(),
        "overPct": round(over_lean * 100, 1),
        "underPct": round(under_lean * 100, 1),
        "confidence": round(confidence, 1),
        "detail": " ".join(detail_parts) if detail_parts else f"Model leans {pick.lower()} vs market total {total_line}.",
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


def _home_field_logit(game: dict[str, Any]) -> float:
    return HOME_FIELD_LOGIT.get(_league_id(game), 0.25)


def _home_field_detail(game: dict[str, Any]) -> str:
    league = _league_id(game)
    venue = game.get("venueName") or "home"
    if league in {"worldcup", "epl"}:
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
    home_form_pct = _last_five_pct(home_form.get("record"))
    away_form_pct = _last_five_pct(away_form.get("record"))
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
                        f"{winner} are {winner_form.get('record')} in their last five ({streak}) while "
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

    if league_config and league_config.id in {"epl", "worldcup"} and home_adv.get("goalDifference") is not None and away_adv.get("goalDifference") is not None:
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


def _build_why_they_win(game: dict[str, Any], reasons: list[dict[str, Any]], predicted_winner: str | None) -> str:
    if not predicted_winner:
        return "Not enough data to explain this pick yet."

    top_reasons = [reason for reason in reasons if reason.get("favors") in {"home", "away"}][:4]
    if not top_reasons:
        league = _league_id(game)
        if league == "worldcup":
            return (
                f"{predicted_winner} are slightly favored in a close match based on form, "
                f"records, and home-field factors."
            )
        return (
            f"{predicted_winner} are slightly favored in a close game based on combined team strength "
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
    home_form_pct = _last_five_pct(home_form.get("record"))
    away_form_pct = _last_five_pct(away_form.get("record"))
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

    model_home = sigmoid(logit)
    true_probs = compute_true_probabilities(
        model_home=model_home,
        enrichment=enrichment,
        league_config=league_config,
        league=league_config.id,
        lines=game.get("lines") or [],
    )

    factors.append(
        {
            "label": "Model probability",
            "detail": (
                f"Data-driven estimate {true_probs['homePct']}% home / {true_probs['awayPct']}% away"
                + (f" / {true_probs['drawPct']}% draw" if true_probs.get("drawPct") is not None else "")
            ),
            "edge": _edge_label(true_probs["home"], true_probs["away"]),
        }
    )

    raw_best = max(
        true_probs["home"],
        true_probs["away"],
        true_probs.get("draw") or 0.0 if league_config.supports_draw else 0.0,
    )
    calibration_confidence = raw_best * 100.0

    home_prob = calibrate_probability(
        clamp(true_probs["home"]),
        league=league_config.id,
        confidence_pct=calibration_confidence,
    )
    away_prob = calibrate_probability(
        clamp(true_probs["away"]),
        league=league_config.id,
        confidence_pct=calibration_confidence,
    )
    draw_prob = (
        calibrate_probability(
            clamp(true_probs.get("draw") or 0.0),
            league=league_config.id,
            confidence_pct=calibration_confidence,
        )
        if league_config.supports_draw
        else 0.0
    )
    prob_total = home_prob + away_prob + draw_prob
    if prob_total > 0:
        home_prob /= prob_total
        away_prob /= prob_total
        draw_prob /= prob_total

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
    why_they_win = (
        _build_why_they_win(game, reasons, predicted_winner)
        if predicted_side != "draw"
        else f"Draw is the most likely result ({round(draw_prob * 100, 1)}%) based on form and matchup data."
    )

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

    result: dict[str, Any] = {
        "predictedWinner": predicted_winner,
        "predictedSide": predicted_side,
        "homeWinPct": home_pct,
        "awayWinPct": away_pct,
        "confidence": round(confidence, 1),
        "confidenceLabel": confidence_label(confidence),
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


def predict_spread(game: dict[str, Any], lines: list[dict[str, Any]], enrichment: dict[str, Any]) -> dict[str, Any] | None:
    """Predict point spread for sports that support spreads (NFL, NBA, NCAAF, etc.)."""
    # Extract spread line from odds
    spread_line = extract_spread_line(lines)
    if spread_line is None:
        return None

    league = _league_id(game)
    
    # For now, use a simple model based on win probability
    # In the future, this could be enhanced with more sophisticated spread modeling
    probs = enrichment.get("probabilities") or {}
    true_p = probs.get("true") or {}
    home_prob = true_p.get("home", 0.5)
    away_prob = true_p.get("away", 0.5)
    
    # Convert win probability to spread estimate
    # A 55% win prob roughly equals a 1-point favorite in NFL
    # A 52% win prob roughly equals a 1-point favorite in NBA
    if league in {"nfl"}:
        points_per_pct = 0.5  # 1% win prob ~ 0.5 points in NFL
    elif league in {"nba", "ncaaf"}:
        points_per_pct = 0.3  # 1% win prob ~ 0.3 points in NBA/NCAAF
    else:
        points_per_pct = 0.2  # Default for other sports
    
    # Calculate model spread from win probability
    prob_diff = home_prob - away_prob
    model_spread = -prob_diff / (points_per_pct / 100)  # Negative because spread is home - away
    
    # Compare model spread to market spread
    edge = model_spread - spread_line
    
    # Determine pick with clearer thresholds
    if abs(edge) < 0.3:
        # Too close to call - lean slightly but low confidence
        pick_side = "push"
        confidence = 50
    elif edge > 0:
        # Model favors home more than market
        pick_side = "home"
        # Higher confidence for larger edges
        confidence = min(55 + abs(edge) * 20, 90)
    else:
        # Model favors away more than market
        pick_side = "away"
        confidence = min(55 + abs(edge) * 20, 90)
    
    # Clear directional language
    if pick_side == "push":
        pick_text = f"Push (no lean)"
    elif pick_side == "home":
        pick_text = f"Home {spread_line:+.1f}"
    else:
        pick_text = f"Away {spread_line:+.1f}"
    
    return {
        "line": spread_line,
        "modelLine": round(model_spread, 1),
        "pick": pick_text,
        "pickSide": pick_side,
        "homePct": round((1 + (spread_line / 100)) * 50, 1),
        "awayPct": round((1 - (spread_line / 100)) * 50, 1),
        "edgePct": round(edge, 1),
        "confidence": confidence,
        "detail": f"Model: {model_spread:+.1f} | Market: {spread_line:+.1f} | Edge: {edge:+.1f} | {pick_text}",
    }
def apply_predictions(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for game in games:
        # First enrich the game
        league = game.get("league", "mlb")
        enriched = enrich_games_with_providers([game], league=league)
        game = enriched[0] if enriched else game
        
        prediction = predict_game(game)
        prediction["publishable"] = is_publishable_pick(prediction)
        
        # Add totals prediction if lines available
        lines = game.get("lines", [])
        enrichment = game.get("enrichment", {})
        if lines:
            total_pred = predict_total(game, lines, enrichment)
            if total_pred:
                prediction["total"] = total_pred
            
            # Add spread prediction for sports that support spreads
            spread_pred = predict_spread(game, lines, enrichment)
            if spread_pred:
                prediction["spread"] = spread_pred
        
        game["prediction"] = prediction

    publishable = [game for game in games if is_publishable_pick(game.get("prediction"))]
    publishable.sort(key=lambda game: game.get("prediction", {}).get("confidence", 0), reverse=True)
    for index, game in enumerate(publishable, start=1):
        game["predictionRank"] = index

    for game in games:
        if game in publishable:
            continue
        game.pop("predictionRank", None)
        prediction = game.get("prediction") or {}
        prediction["publishable"] = False
        game["prediction"] = prediction

    return games
