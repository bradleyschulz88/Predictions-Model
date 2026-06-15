"""Win probability model and human-readable reasoning for scheduled games."""

from __future__ import annotations

import math
import re
from typing import Any

from sports_config import get_league

HOME_FIELD_LOGIT = {
    "mlb": 0.28,
    "worldcup": 0.35,
    "afl": 0.22,
}


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


def _home_field_logit(game: dict[str, Any]) -> float:
    return HOME_FIELD_LOGIT.get(_league_id(game), 0.25)


def _home_field_detail(game: dict[str, Any]) -> str:
    league = _league_id(game)
    venue = game.get("venueName") or "home"
    if league == "worldcup":
        return f"{game.get('homeTeam')} play at {venue}, where host nations and familiar conditions can help."
    if league == "afl":
        return f"{game.get('homeTeam')} have home-ground advantage at {venue}."
    return f"{game.get('homeTeam')} play at {venue}, where MLB teams historically win more often."


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

    home_prob = clamp(home_prob)
    away_prob = 1.0 - home_prob
    predicted_side = "home" if home_prob >= away_prob else "away"
    predicted_winner = _team_by_side(game, predicted_side)
    confidence = max(home_prob, away_prob)

    reasons = _build_reasons(
        game,
        predicted_side=predicted_side,
        home_prob=home_prob,
        away_prob=away_prob,
        enrichment=enrichment,
    )
    why_they_win = _build_why_they_win(game, reasons, predicted_winner)

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
        "homeWinPct": round(home_prob * 100, 1),
        "awayWinPct": round(away_prob * 100, 1),
        "confidence": round(confidence * 100, 1),
        "outcomeLabel": f"{predicted_winner} to win",
        "whyTheyWin": why_they_win,
        "reasons": reasons,
        "factors": factors,
        "dataSources": sorted(set(data_sources)),
    }
    if league_config.supports_draw and market_draw is not None:
        result["drawWinPct"] = round(market_draw * 100, 1)
    return result


def apply_predictions(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for game in games:
        game["prediction"] = predict_game(game)

    games.sort(key=lambda game: game.get("prediction", {}).get("confidence", 0), reverse=True)
    for index, game in enumerate(games, start=1):
        game["predictionRank"] = index
    return games
