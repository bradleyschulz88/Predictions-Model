"""Win probability model and human-readable reasoning for scheduled games."""

from __future__ import annotations

import math
import re
from typing import Any

from sports_config import get_league

HOME_FIELD_LOGIT = {
    "mlb": 0.28,
    "nfl": 0.32,
    "nba": 0.24,
    "worldcup": 0.35,
    "epl": 0.30,
    "afl": 0.22,
}

DEFAULT_DRAW_PROB = 0.26


def parse_record(summary: str | None) -> tuple[int, ...] | None:
    if not summary:
        return None
    match = re.match(r"(\d+)-(\d+)(?:-(\d+))?", summary.strip())
    if not match:
        return None
    parts = [int(match.group(1)), int(match.group(2))]
    if match.group(3) is not None:
        parts.append(int(match.group(3)))
    return tuple(parts)


def win_pct_from_record(summary: str | None, default: float = 0.5) -> float:
    parsed = parse_record(summary)
    if not parsed:
        return default
    if len(parsed) == 3:
        wins, draws, losses = parsed
        total = wins + draws + losses
        return (wins + 0.5 * draws) / total if total else default
    wins, losses = parsed
    total = wins + losses
    return wins / total if total else default


def format_win_pct(summary: str | None) -> str:
    pct = win_pct_from_record(summary)
    return f"{pct * 100:.1f}%"


def american_odds_to_implied(odds: int | float) -> float:
    value = float(odds)
    if value < 0:
        return abs(value) / (abs(value) + 100.0)
    return 100.0 / (value + 100.0)


def extract_moneyline_probs(
    lines: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    for line in lines:
        view_type = line.get("viewType") or ""
        if "MoneyLine" not in view_type:
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        home_ml = current.get("home")
        away_ml = current.get("away")
        draw_ml = current.get("draw")
        if home_ml is None or away_ml is None:
            continue
        home_imp = american_odds_to_implied(home_ml)
        away_imp = american_odds_to_implied(away_ml)
        draw_imp = american_odds_to_implied(draw_ml) if draw_ml is not None else 0.0
        total = home_imp + away_imp + draw_imp
        if total <= 0:
            continue
        return home_imp / total, away_imp / total, (draw_imp / total if draw_imp else None)
    return None, None, None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def clamp(value: float, low: float = 0.05, high: float = 0.95) -> float:
    return max(low, min(high, value))


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


def confidence_label(confidence_pct: float) -> str:
    if confidence_pct >= 65:
        return "Strong pick"
    if confidence_pct >= 55:
        return "Lean"
    return "Coin flip"


def _injury_logit_adjustment(enrichment: dict[str, Any]) -> float:
    home_major = len(enrichment.get("homeMajorInjuries") or [])
    away_major = len(enrichment.get("awayMajorInjuries") or [])
    return (away_major - home_major) * 0.18


def _advanced_logit_adjustment(enrichment: dict[str, Any]) -> float:
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
            over_lean += 0.12
            detail_parts.append(f"Recent scoring pace ({pace:.1f}) runs hot vs the {total_line} line.")
        elif pace <= total_line - 0.8:
            over_lean -= 0.12
            detail_parts.append(f"Recent scoring pace ({pace:.1f}) runs cool vs the {total_line} line.")

    home_pitcher = game.get("homePitcher") or {}
    away_pitcher = game.get("awayPitcher") or {}
    if home_pitcher.get("era") is not None and away_pitcher.get("era") is not None:
        avg_era = (home_pitcher["era"] + away_pitcher["era"]) / 2
        if avg_era <= 3.6:
            over_lean -= 0.08
            detail_parts.append(f"Strong pitching matchup (avg ERA {avg_era:.2f}) favors the under.")
        elif avg_era >= 4.6:
            over_lean += 0.08
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
            over_lean += 0.06
            detail_parts.append(f"Season scoring pace ({combined:.1f} combined R/G) leans over.")
        elif combined <= total_line - 1.5:
            over_lean -= 0.06
            detail_parts.append(f"Season scoring pace ({combined:.1f} combined R/G) leans under.")

    if league in {"nba", "afl"}:
        home_overall = win_pct_from_record(game.get("homeRecord"))
        away_overall = win_pct_from_record(game.get("awayRecord"))
        if home_overall + away_overall > 1.05:
            over_lean += 0.05
            detail_parts.append("Both teams have strong records — higher-scoring game possible.")

    over_lean = clamp(over_lean, 0.35, 0.65)
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
    if league == "nba":
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

    market_home, market_away, market_draw = extract_moneyline_probs(game.get("lines") or [])
    if market_home is not None and market_away is not None:
        market_side = "home" if market_home >= market_away else "away"
        if market_side == predicted_side:
            draw_note = f", draw {market_draw * 100:.1f}%" if market_draw is not None else ""
            source = "ESPN" if game.get("oddsSource") == "espn" else "SportsBookReview"
            reasons.append(
                {
                    "title": "Betting market leans the same way",
                    "detail": (
                        f"Moneylines imply {market_home * 100:.1f}% for {game.get('homeTeam')} "
                        f"and {market_away * 100:.1f}% for {game.get('awayTeam')}{draw_note}."
                    ),
                    "impact": "high",
                    "favors": predicted_side,
                    "source": source,
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
                            f"{game.get('homeTeam')} {home_adv['runDifferential']:+d} vs "
                            f"{game.get('awayTeam')} {away_adv['runDifferential']:+d} (MLB.com)."
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
                        f"{game.get('homeTeam')} GD {home_adv['goalDifference']:+d} vs "
                        f"{game.get('awayTeam')} {away_adv['goalDifference']:+d}."
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
        era_diff = away_era - home_era
        logit += era_diff * 0.38
        factors.append(
            {
                "label": "Starting pitching",
                "detail": (
                    f"{home_pitcher.get('name') or 'Home SP'} ({home_era:.2f} ERA) vs "
                    f"{away_pitcher.get('name') or 'Away SP'} ({away_era:.2f} ERA)"
                ),
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

    injury_adj = _injury_logit_adjustment(enrichment)
    if injury_adj:
        logit += injury_adj
        home_major = len(enrichment.get("homeMajorInjuries") or [])
        away_major = len(enrichment.get("awayMajorInjuries") or [])
        factors.append(
            {
                "label": "Injury impact",
                "detail": f"Major injuries: home {home_major} vs away {away_major}",
                "edge": "home" if injury_adj > 0 else "away" if injury_adj < 0 else "even",
            }
        )

    advanced_adj = _advanced_logit_adjustment(enrichment)
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
    market_home, market_away, market_draw = extract_moneyline_probs(game.get("lines") or [])

    components: list[tuple[float, float]] = [(model_home, 0.40)]
    if espn_home is not None and espn_away is not None:
        espn_total = espn_home + espn_away
        if espn_total > 0:
            components.append((espn_home / espn_total, 0.25))
    if market_home is not None and market_away is not None:
        components.append((market_home, 0.35))
        market_detail = f"Moneyline implied {market_home * 100:.1f}% home / {market_away * 100:.1f}% away"
        if market_draw is not None:
            market_detail += f" / {market_draw * 100:.1f}% draw"
        factors.append(
            {
                "label": "Betting market",
                "detail": market_detail,
                "edge": _edge_label(market_home, market_away),
            }
        )
    else:
        factors.append(
            {
                "label": "Betting market",
                "detail": "No moneyline odds available yet",
                "edge": "even",
            }
        )

    weight_total = sum(weight for _, weight in components)
    home_prob = sum(prob * weight for prob, weight in components) / weight_total

    draw_prob = market_draw if market_draw is not None else (DEFAULT_DRAW_PROB if league_config.supports_draw else 0.0)
    home_prob = clamp(home_prob)
    away_prob = 1.0 - home_prob

    if league_config.supports_draw and draw_prob:
        draw_prob = clamp(draw_prob, 0.08, 0.40)
        scale = 1.0 - draw_prob
        home_prob = home_prob * scale
        away_prob = away_prob * scale
    else:
        draw_prob = 0.0

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
                "detail": f"Model and market imply a {round(draw_prob * 100, 1)}% chance of a draw.",
                "impact": "high",
                "favors": "even",
                "source": "Model",
            },
        )
    why_they_win = (
        _build_why_they_win(game, reasons, predicted_winner)
        if predicted_side != "draw"
        else f"Draw is the most likely result ({round(draw_prob * 100, 1)}%) based on market and form."
    )

    market_edge = _model_market_edge(
        predicted_side=predicted_side if predicted_side in {"home", "away"} else "home",
        home_prob=home_prob,
        away_prob=away_prob,
        market_home=market_home,
        market_away=market_away,
    )
    if market_edge and abs(market_edge["edgePct"]) >= 3 and predicted_side in {"home", "away"}:
        reasons.insert(
            0,
            {
                "title": "Model vs market edge",
                "detail": (
                    f"Model {market_edge['modelPct']}% vs market {market_edge['marketPct']}% "
                    f"({market_edge['edgeLabel']})."
                ),
                "impact": "high",
                "favors": predicted_side,
            },
        )

    total_prediction = predict_total(game, game.get("lines") or [], enrichment)

    data_sources = ["ESPN scoreboard"]
    if enrichment:
        data_sources.extend(enrichment.get("sources") or [])
    if game.get("lines"):
        if game.get("oddsSource") == "espn":
            data_sources.append("ESPN betting odds")
        elif game.get("oddsSource") == "sbr":
            data_sources.append("SportsBookReview odds")
        else:
            data_sources.append("Betting odds")

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
        "modelEdge": market_edge,
    }
    if league_config.supports_draw and draw_prob:
        result["drawWinPct"] = round(draw_prob * 100, 1)
    if total_prediction:
        result["totalPick"] = total_prediction
    return result


def apply_predictions(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for game in games:
        game["prediction"] = predict_game(game)

    games.sort(key=lambda game: game.get("prediction", {}).get("confidence", 0), reverse=True)
    for index, game in enumerate(games, start=1):
        game["predictionRank"] = index
    return games
