"""Fetch extra matchup context from ESPN's game summary API."""

from __future__ import annotations

import json
import time
from typing import Any

from espn_client import build_summary_url
from mlb_cache import ENRICHMENT_CACHE
from sbr_client import SBRClientError, get_text
from sports_config import LeagueConfig, get_league

SIGNIFICANT_INJURY_KEYWORDS = ("IL", "Out", "Suspended", "Day-To-Day", "Doubtful")
MAJOR_INJURY_KEYWORDS = ("60-Day", "15-Day", "7-Day", " IL", "Out", "Suspended")
MINOR_INJURY_EXCLUDE = ("Paternity", "Personal", "Bereavement")


class ESPNEnrichmentError(SBRClientError):
    """Error while fetching or parsing ESPN summary data."""


def fetch_event_summary(
    event_id: str | int,
    *,
    league: LeagueConfig | str = "mlb",
    retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    url = build_summary_url(league, event_id)
    text = get_text(url, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ESPNEnrichmentError(f"Invalid JSON from ESPN summary: {exc}") from exc
    if not isinstance(data, dict):
        raise ESPNEnrichmentError("ESPN summary response is not an object")
    return data


def _parse_american_odds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("+", "")
    try:
        return int(text)
    except ValueError:
        return None


def _moneyline_side(block: dict[str, Any] | None) -> int | None:
    if not block:
        return None
    close = block.get("close") or block.get("open") or {}
    return _parse_american_odds(close.get("odds"))


def parse_espn_odds(summary: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for block in summary.get("odds") or []:
        provider = (block.get("provider") or {}).get("name") or "ESPN"
        moneyline = block.get("moneyline") or {}
        current_ml: dict[str, int] = {}
        opening_ml: dict[str, int] = {}

        for side in ("home", "away", "draw"):
            side_block = moneyline.get(side)
            if not side_block:
                continue
            current = _moneyline_side(side_block)
            if current is not None:
                current_ml[side] = current
            open_block = side_block.get("open") or {}
            open_odds = _parse_american_odds(open_block.get("odds"))
            if open_odds is not None:
                opening_ml[side] = open_odds

        if current_ml:
            lines.append(
                {
                    "sportsbook": provider,
                    "viewType": "MoneyLine",
                    "currentLine": current_ml,
                    "openingLine": opening_ml or None,
                }
            )

        spread = block.get("pointSpread") or {}
        spread_current: dict[str, str | int] = {}
        for side in ("home", "away"):
            side_block = spread.get(side) or {}
            close = side_block.get("close") or side_block.get("open") or {}
            line = close.get("line")
            odds = close.get("odds")
            if line is not None:
                spread_current[side] = f"{line} ({odds})" if odds is not None else str(line)
        if spread_current:
            lines.append(
                {
                    "sportsbook": provider,
                    "viewType": "Spread",
                    "currentLine": spread_current,
                }
            )

        total = block.get("total") or {}
        total_current: dict[str, str] = {}
        for side in ("over", "under"):
            side_block = total.get(side) or {}
            close = side_block.get("close") or side_block.get("open") or {}
            line = close.get("line")
            odds = close.get("odds")
            if line is not None:
                total_current[side] = f"{line} ({odds})" if odds is not None else str(line)
        if total_current:
            lines.append(
                {
                    "sportsbook": provider,
                    "viewType": "Total",
                    "currentLine": total_current,
                }
            )

    return lines


def _last_five_for_team(last_five_games: list[dict[str, Any]] | None, team_name: str | None) -> dict[str, Any]:
    for block in last_five_games or []:
        if (block.get("team") or {}).get("displayName") != team_name:
            continue
        events = block.get("events") or []
        wins = sum(1 for event in events if event.get("gameResult") == "W")
        draws = sum(1 for event in events if event.get("gameResult") == "D")
        losses = sum(1 for event in events if event.get("gameResult") == "L")
        if draws:
            record = f"{wins}-{draws}-{losses}"
        else:
            record = f"{wins}-{losses}"
        return {
            "record": record,
            "results": [event.get("gameResult") for event in events[:5] if event.get("gameResult")],
            "games": [
                {
                    "result": event.get("gameResult"),
                    "score": event.get("score"),
                    "opponent": (event.get("opponent") or {}).get("displayName"),
                }
                for event in events[:5]
            ],
        }
    return {"record": None, "results": [], "games": []}


def _position_abbrev(entry: dict[str, Any]) -> str | None:
    position = entry.get("position")
    if isinstance(position, dict):
        return position.get("abbreviation") or position.get("name")
    if isinstance(position, str):
        return position
    athlete = entry.get("athlete") or {}
    athlete_position = athlete.get("position")
    if isinstance(athlete_position, dict):
        return athlete_position.get("abbreviation") or athlete_position.get("name")
    return None


def _parse_lineup_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batters: list[dict[str, Any]] = []
    for entry in entries:
        athlete = entry.get("athlete") or {}
        name = athlete.get("displayName") or athlete.get("fullName")
        if not name:
            continue
        batters.append(
            {
                "order": entry.get("batOrder"),
                "name": name,
                "position": _position_abbrev(entry),
            }
        )

    batters.sort(key=lambda batter: batter.get("order") if batter.get("order") is not None else 99)
    numbered = [batter for batter in batters if batter.get("order") is not None]
    return numbered[:9] if numbered else batters[:9]


def _lineup_from_rosters(rosters: list[dict[str, Any]] | None, team_name: str | None) -> list[dict[str, Any]]:
    for roster_block in rosters or []:
        if (roster_block.get("team") or {}).get("displayName") != team_name:
            continue
        entries = roster_block.get("roster") or []
        lineup = _parse_lineup_entries(entries)
        if lineup:
            return lineup
    return []


def _lineup_from_boxscore(boxscore: dict[str, Any] | None, team_name: str | None) -> list[dict[str, Any]]:
    for player_group in (boxscore or {}).get("players") or []:
        if (player_group.get("team") or {}).get("displayName") != team_name:
            continue
        for stat_group in player_group.get("statistics") or []:
            athletes = stat_group.get("athletes") or []
            lineup = _parse_lineup_entries(athletes)
            if lineup:
                return lineup
    return []


def _leaders_fallback(leaders_block: list[dict[str, Any]] | None, team_name: str | None) -> list[dict[str, Any]]:
    hitters: list[dict[str, Any]] = []
    for block in leaders_block or []:
        if (block.get("team") or {}).get("displayName") != team_name:
            continue
        for category in block.get("leaders") or []:
            category_name = category.get("displayName") or category.get("name")
            for leader in category.get("leaders") or []:
                athlete = leader.get("athlete") or {}
                name = athlete.get("displayName")
                if not name:
                    continue
                hitters.append(
                    {
                        "name": name,
                        "position": category_name,
                        "statLine": leader.get("displayValue"),
                    }
                )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hitter in hitters:
        if hitter["name"] in seen:
            continue
        seen.add(hitter["name"])
        deduped.append(hitter)
        if len(deduped) >= 6:
            break
    return deduped


def _build_lineup_payload(
    summary: dict[str, Any],
    *,
    team_name: str | None,
    lineup_label: str,
) -> dict[str, Any]:
    lineup = _lineup_from_rosters(summary.get("rosters"), team_name)
    source = "espn_roster"
    if not lineup:
        lineup = _lineup_from_boxscore(summary.get("boxscore"), team_name)
        source = "espn_boxscore"

    if lineup:
        return {
            "status": "confirmed",
            "source": source,
            "note": f"Starting lineup from ESPN.",
            "batters": lineup,
        }

    fallback = _leaders_fallback(summary.get("leaders"), team_name)
    if fallback:
        return {
            "status": "projected",
            "source": "espn_leaders",
            "note": f"{lineup_label} not posted yet. Key players to watch:",
            "batters": fallback,
        }

    return {
        "status": "unavailable",
        "source": None,
        "note": f"{lineup_label} not available yet.",
        "batters": [],
    }


def _injury_is_major(status: str, detail: str) -> bool:
    combined = f"{status} {detail}"
    if any(keyword in combined for keyword in MINOR_INJURY_EXCLUDE):
        return False
    return any(keyword in combined for keyword in MAJOR_INJURY_KEYWORDS)


def _parse_injuries(injuries_block: list[dict[str, Any]] | None, team_name: str | None) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for team_block in injuries_block or []:
        if (team_block.get("team") or {}).get("displayName") != team_name:
            continue
        for injury in team_block.get("injuries") or []:
            athlete = injury.get("athlete") or {}
            status = injury.get("status") or injury.get("type") or "Unknown"
            details = injury.get("details") or {}
            detail_text = details.get("detail") or details.get("type") or injury.get("longComment") or status
            side = details.get("side")
            if side and side not in detail_text:
                detail_text = f"{detail_text} ({side})"
            return_date = details.get("returnDate") or injury.get("returnDate")
            parsed.append(
                {
                    "player": athlete.get("displayName") or "Unknown",
                    "status": status,
                    "detail": detail_text,
                    "returnDate": return_date,
                }
            )
    return parsed


def _major_injuries(injuries: list[dict[str, str]]) -> list[dict[str, str]]:
    major: list[dict[str, str]] = []
    for injury in injuries:
        if _injury_is_major(injury.get("status", ""), injury.get("detail", "")):
            major.append(injury)
    return major[:8]


def _key_injuries(injuries: list[dict[str, str]]) -> list[str]:
    keys: list[str] = []
    for injury in injuries:
        status = injury.get("status") or ""
        if any(keyword in status for keyword in SIGNIFICANT_INJURY_KEYWORDS):
            keys.append(f"{injury['player']} ({status})")
    return keys[:5]


def _regular_season_series(season_series: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    for series in season_series or []:
        title = series.get("title") or ""
        if series.get("type") == "season" or "Regular Season" in title:
            return {
                "title": title,
                "summary": series.get("summary"),
                "seriesScore": series.get("seriesScore"),
            }
    return None


def _head_to_head_series(head_to_head: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not head_to_head:
        return None
    summaries: list[str] = []
    for block in head_to_head:
        team_name = (block.get("team") or {}).get("displayName")
        events = block.get("events") or []
        if not events:
            continue
        wins = sum(1 for event in events if event.get("gameResult") == "W")
        losses = sum(1 for event in events if event.get("gameResult") == "L")
        draws = sum(1 for event in events if event.get("gameResult") == "D")
        if draws:
            summaries.append(f"{team_name} {wins}-{draws}-{losses} in recent meetings")
        else:
            summaries.append(f"{team_name} {wins}-{losses} in recent meetings")
    if not summaries:
        return None
    return {
        "title": "Head-to-head",
        "summary": "; ".join(summaries),
        "seriesScore": None,
    }


def _parse_weather(game_info: dict[str, Any] | None) -> str | None:
    weather = (game_info or {}).get("weather")
    if not isinstance(weather, dict):
        return None
    temp = weather.get("temperature")
    precip = weather.get("precipitation")
    if temp is None:
        return None
    if precip is None:
        return f"{temp}°F"
    return f"{temp}°F, {precip}% precipitation"


def parse_event_enrichment(
    summary: dict[str, Any],
    *,
    home_team: str | None,
    away_team: str | None,
    league: LeagueConfig | str = "mlb",
) -> dict[str, Any]:
    league_config = get_league(league) if isinstance(league, str) else league
    predictor = summary.get("predictor") or {}
    home_predictor = predictor.get("homeTeam") or {}
    away_predictor = predictor.get("awayTeam") or {}

    home_last_five = _last_five_for_team(summary.get("lastFiveGames"), home_team)
    away_last_five = _last_five_for_team(summary.get("lastFiveGames"), away_team)
    home_injuries = _parse_injuries(summary.get("injuries"), home_team)
    away_injuries = _parse_injuries(summary.get("injuries"), away_team)
    series = _regular_season_series(summary.get("seasonseries")) or _head_to_head_series(summary.get("headToHeadGames"))
    home_lineup = _build_lineup_payload(summary, team_name=home_team, lineup_label=league_config.lineup_label)
    away_lineup = _build_lineup_payload(summary, team_name=away_team, lineup_label=league_config.lineup_label)
    home_major_injuries = _major_injuries(home_injuries)
    away_major_injuries = _major_injuries(away_injuries)
    espn_odds = parse_espn_odds(summary)

    sources = [
        "ESPN game summary",
        "ESPN last-five form",
        "ESPN injury report",
        league_config.lineup_label,
    ]
    if predictor:
        sources.append("ESPN Matchup Predictor")
    if series:
        sources.append("ESPN head-to-head")
    if espn_odds:
        sources.append("ESPN betting odds")

    return {
        "espnPredictorHome": _to_float(home_predictor.get("gameProjection")),
        "espnPredictorAway": _to_float(away_predictor.get("gameProjection")),
        "homeLastFive": home_last_five,
        "awayLastFive": away_last_five,
        "seasonSeries": series,
        "weather": _parse_weather(summary.get("gameInfo")),
        "homeInjuries": home_injuries,
        "awayInjuries": away_injuries,
        "homeMajorInjuries": home_major_injuries,
        "awayMajorInjuries": away_major_injuries,
        "homeKeyInjuries": _key_injuries(home_injuries),
        "awayKeyInjuries": _key_injuries(away_injuries),
        "homeLineup": home_lineup,
        "awayLineup": away_lineup,
        "espnOdds": espn_odds,
        "sources": sources,
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_espn_odds(game: dict[str, Any], enrichment: dict[str, Any]) -> None:
    espn_odds = enrichment.get("espnOdds") or []
    if not espn_odds:
        return
    existing = game.get("lines") or []
    if existing:
        game["lines"] = existing + espn_odds
    else:
        game["lines"] = espn_odds
    game["oddsSource"] = game.get("oddsSource") or "espn"
    view_types = game.get("viewTypes") or []
    for line in espn_odds:
        view_type = line.get("viewType")
        if view_type and view_type not in view_types:
            view_types.append(view_type)
    game["viewTypes"] = view_types


def ensure_espn_odds_on_games(games: list[dict[str, Any]]) -> None:
    """Attach ESPN summary odds when a game has no usable moneyline lines."""
    from mlb_predictions import has_moneyline_lines

    for game in games:
        if has_moneyline_lines(game.get("lines") or []):
            continue
        enrichment = game.get("enrichment") or {}
        if enrichment.get("espnOdds"):
            _merge_espn_odds(game, enrichment)


def enrich_game(
    game: dict[str, Any],
    *,
    retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
    summary_fixture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    league = game.get("league") or "mlb"
    event_id = game.get("eventId")
    if summary_fixture is not None:
        game["enrichment"] = parse_event_enrichment(
            summary_fixture,
            home_team=game.get("homeTeam"),
            away_team=game.get("awayTeam"),
            league=league,
        )
        _merge_espn_odds(game, game["enrichment"])
        _copy_enrichment_to_game(game)
        return game

    if not event_id:
        game["enrichment"] = {}
        _copy_enrichment_to_game(game)
        return game

    try:
        cache_key = f"summary:{league}:{event_id}"
        cached_summary = ENRICHMENT_CACHE.get(cache_key)
        if cached_summary is not None:
            summary = cached_summary
        else:
            summary = fetch_event_summary(
                event_id,
                league=league,
                retries=retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            )
            ENRICHMENT_CACHE.set(cache_key, summary)
        game["enrichment"] = parse_event_enrichment(
            summary,
            home_team=game.get("homeTeam"),
            away_team=game.get("awayTeam"),
            league=league,
        )
        _merge_espn_odds(game, game["enrichment"])
    except SBRClientError:
        game["enrichment"] = {}

    _copy_enrichment_to_game(game)
    return game


def _copy_enrichment_to_game(game: dict[str, Any]) -> None:
    enrichment = game.get("enrichment") or {}
    game["homeLineup"] = enrichment.get("homeLineup")
    game["awayLineup"] = enrichment.get("awayLineup")
    game["homeMajorInjuries"] = enrichment.get("homeMajorInjuries") or []
    game["awayMajorInjuries"] = enrichment.get("awayMajorInjuries") or []


def enrich_games(
    games: list[dict[str, Any]],
    *,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
    request_delay: float = 0.05,
) -> list[dict[str, Any]]:
    for index, game in enumerate(games):
        enrich_game(game, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl)
        if index + 1 < len(games):
            time.sleep(request_delay)
    return games
