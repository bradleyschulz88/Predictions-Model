"""Transform schedule and odds data into dashboard payloads."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_providers.utils import team_match_score
from espn_client import ESPNClientError, fetch_scoreboard, parse_scoreboard
from espn_enrichment import enrich_game, enrich_games, ensure_espn_odds_on_games
from data_providers import enrich_games_with_providers
from mlb_predictions import apply_predictions, is_publishable_pick
from data_providers.schedule_advanced import clear_rolling_schedule_cache, fetch_rolling_schedule_games
from schedule_dates import default_game_date, get_schedule_timezone
from sbr_client import SBRClientError, build_odds_url, get_game_rows, get_page_props
from sports_config import LEAGUES, get_league


def game_summary(row: dict[str, Any]) -> dict[str, Any]:
    game_view = row.get("gameView") or {}
    away = (game_view.get("awayTeam") or {}).get("fullName")
    home = (game_view.get("homeTeam") or {}).get("fullName")
    return {
        "startDate": game_view.get("startDate"),
        "awayTeam": away,
        "homeTeam": home,
        "matchup": f"{away} @ {home}" if away and home else None,
        "gameStatusText": game_view.get("gameStatusText"),
        "venueName": game_view.get("venueName"),
        "source": "sbr",
        "viewTypes": [],
        "lines": [],
    }


def collect_view_types(row: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for odds_view in row.get("oddsViews") or []:
        if not odds_view:
            continue
        view_type = odds_view.get("viewType")
        if view_type and view_type not in seen:
            seen.append(view_type)
    return seen


def matches_view_filter(view_type: str | None, view_filter: str | None) -> bool:
    if not view_filter:
        return True
    if not view_type:
        return False
    parts = [part.strip() for part in view_filter.split("|") if part.strip()]
    return any(part in view_type for part in parts)


def collect_odds_lines(row: dict[str, Any], *, view_filter: str | None = None) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for odds_view in row.get("oddsViews") or []:
        if not odds_view:
            continue
        view_type = odds_view.get("viewType")
        if not matches_view_filter(view_type, view_filter):
            continue
        lines.append(
            {
                "sportsbook": odds_view.get("sportsbook"),
                "viewType": view_type,
                "openingLine": odds_view.get("openingLine"),
                "currentLine": odds_view.get("currentLine"),
            }
        )
    return lines


def normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return cleaned


def matchup_key(away_team: str | None, home_team: str | None) -> str:
    return f"{normalize_team_name(away_team)}|{normalize_team_name(home_team)}"


def _find_sbr_odds_match(
    away_team: str | None,
    home_team: str | None,
    odds_by_matchup: dict[str, list[dict[str, Any]]],
    view_types_by_matchup: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[str]] | None:
    key = matchup_key(away_team, home_team)
    if key in odds_by_matchup:
        return odds_by_matchup[key], view_types_by_matchup.get(key, [])

    best_key: str | None = None
    best_score = 0.0
    for candidate_key in odds_by_matchup:
        sbr_away, sbr_home = candidate_key.split("|", 1)
        away_score = team_match_score(away_team, sbr_away)
        home_score = team_match_score(home_team, sbr_home)
        if away_score < 0.55 or home_score < 0.55:
            continue
        combined = (away_score + home_score) / 2
        if combined > best_score:
            best_score = combined
            best_key = candidate_key

    if not best_key:
        return None
    return odds_by_matchup[best_key], view_types_by_matchup.get(best_key, [])


def _attach_sbr_lines(game: dict[str, Any], lines: list[dict[str, Any]], view_types: list[str]) -> None:
    game["lines"] = lines
    game["viewTypes"] = view_types
    game["oddsSource"] = "sbr"


def merge_sbr_odds_into_games(
    games: list[dict[str, Any]],
    *,
    league: str = "mlb",
    date_value: str,
    view_filter: str = "Spread|MoneyLine|Total",
    retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
) -> None:
    league_config = get_league(league)
    odds_slug = league_config.sbr_odds_slug
    if not odds_slug:
        return

    try:
        page_props = get_page_props(
            build_odds_url(date_value, odds_slug=odds_slug),
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )
    except SBRClientError:
        return

    odds_by_matchup: dict[str, list[dict[str, Any]]] = {}
    view_types_by_matchup: dict[str, list[str]] = {}

    for row in get_game_rows(page_props):
        summary = game_summary(row)
        key = matchup_key(summary.get("awayTeam"), summary.get("homeTeam"))
        odds_by_matchup[key] = collect_odds_lines(row, view_filter=view_filter)
        view_types_by_matchup[key] = collect_view_types(row)

    for game in games:
        matched = _find_sbr_odds_match(
            game.get("awayTeam"),
            game.get("homeTeam"),
            odds_by_matchup,
            view_types_by_matchup,
        )
        if matched:
            _attach_sbr_lines(game, matched[0], matched[1])


def load_fixture_data(fixture_path: str | Path) -> dict[str, Any]:
    with open(fixture_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SBRClientError("Fixture root must be an object")
    return data


def load_page_props_from_file(fixture_path: str | Path) -> dict[str, Any]:
    data = load_fixture_data(fixture_path)
    if "props" in data and "pageProps" in data.get("props", {}):
        return data["props"]["pageProps"]
    if "pageProps" in data:
        return data["pageProps"]
    if "events" in data:
        raise SBRClientError("Fixture looks like ESPN scoreboard; use source=espn")
    if isinstance(data, dict):
        return data
    raise SBRClientError("Fixture does not contain pageProps")


def load_espn_scoreboard_from_file(fixture_path: str | Path) -> dict[str, Any]:
    data = load_fixture_data(fixture_path)
    if "events" in data:
        return data
    raise ESPNClientError("Fixture does not contain ESPN scoreboard events")


def build_dashboard_payload_from_sbr(
    page_props: dict[str, Any],
    *,
    url: str,
    view_filter: str | None = "Spread|MoneyLine|Total",
) -> dict[str, Any]:
    rows = get_game_rows(page_props)
    games: list[dict[str, Any]] = []

    for row in rows:
        item = game_summary(row)
        item["viewTypes"] = collect_view_types(row)
        item["lines"] = collect_odds_lines(row, view_filter=view_filter)
        games.append(item)

    return finalize_dashboard_payload(games, url=url, source="sbr", league="mlb")


def build_dashboard_payload_from_espn_games(
    games: list[dict[str, Any]],
    *,
    url: str,
    league: str = "mlb",
) -> dict[str, Any]:
    return finalize_dashboard_payload(games, url=url, source="espn", league=league)


def strip_betting_lines_for_display(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove sportsbook line data from client-facing payloads."""
    cleaned = dict(payload)
    cleaned.pop("sportsbooks", None)
    cleaned.pop("sportsbookCount", None)
    display_games: list[dict[str, Any]] = []
    for game in payload.get("games") or []:
        display_game = dict(game)
        display_game.pop("lines", None)
        display_game.pop("oddsSource", None)
        display_games.append(display_game)
    cleaned["games"] = display_games
    return cleaned


def finalize_dashboard_payload(
    games: list[dict[str, Any]],
    *,
    url: str,
    source: str,
    league: str = "mlb",
) -> dict[str, Any]:
    league_config = get_league(league)
    games = apply_predictions(games)

    statuses: dict[str, int] = {}
    sportsbooks: set[str] = set()
    for game in games:
        status = game.get("gameStatusText") or "Unknown"
        statuses[status] = statuses.get(status, 0) + 1
        for line in game.get("lines") or []:
            book = line.get("sportsbook")
            if book:
                sportsbooks.add(book)

    top_game = next((game for game in games if game.get("predictionRank") == 1), None)
    if top_game is None:
        publishable = [game for game in games if is_publishable_pick(game.get("prediction"))]
        if publishable:
            top_game = max(
                publishable,
                key=lambda item: (item.get("prediction") or {}).get("confidence") or 0,
            )
    top_pick = (top_game.get("prediction") or {}).get("outcomeLabel") if top_game else None

    return {
        "url": url,
        "source": source,
        "league": league_config.id,
        "leagueLabel": league_config.label,
        "leagues": [
            {"id": item.id, "label": item.label, "shortLabel": item.short_label}
            for item in LEAGUES.values()
        ],
        "scheduleDate": url.split("dates=")[-1] if "dates=" in url else None,
        "gameCount": len(games),
        "statusCounts": statuses,
        "sportsbookCount": len(sportsbooks),
        "sportsbooks": sorted(sportsbooks),
        "topPick": top_pick,
        "games": games,
        "fetchedAt": datetime.now(timezone.utc).isoformat(),
    }


def fetch_dashboard_data(
    *,
    league: str = "mlb",
    date: str | None = None,
    source: str = "espn",
    fixture: str | Path | None = None,
    view_filter: str = "Spread|MoneyLine|Total",
    include_odds: bool = True,
    include_enrichment: bool = True,
    retries: int = 3,
    retry_delay: float = 1.0,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    league_config = get_league(league)
    date_value = date or default_game_date(league)

    if fixture:
        fixture_path = Path(fixture)
        data = load_fixture_data(fixture_path)
        if "events" in data:
            games = parse_scoreboard(data, league=league)
            if include_enrichment:
                _attach_offline_enrichment_samples(games)
            if include_odds and league_config.supports_sbr_odds:
                merge_sbr_odds_into_games(
                    games,
                    league=league,
                    date_value=date_value,
                    view_filter=view_filter,
                    retries=retries,
                    retry_delay=retry_delay,
                    verify_ssl=verify_ssl,
                )
            # Also load odds from fixture if available (for offline/testing)
            if include_odds:
                odds_fixture_path = fixture_path.parent / "odds_page.json"
                if odds_fixture_path.exists():
                    odds_data = load_fixture_data(odds_fixture_path)
                    odds_rows = odds_data.get("props", {}).get("pageProps", {}).get("oddsTables", [{}])[0].get("oddsTableModel", {}).get("gameRows", [])
                    if odds_rows:
                        for row in odds_rows:
                            lines = collect_odds_lines(row)
                            if lines:
                                matched = _find_sbr_odds_match(
                                    row["gameView"]["awayTeam"]["fullName"],
                                    row["gameView"]["homeTeam"]["fullName"],
                                    {matchup_key(row["gameView"]["awayTeam"]["fullName"], row["gameView"]["homeTeam"]["fullName"]): lines},
                                    {matchup_key(row["gameView"]["awayTeam"]["fullName"], row["gameView"]["homeTeam"]["fullName"]): []}
                                )
                                if matched:
                                    for game in games:
                                        if game.get("awayTeam") == row["gameView"]["awayTeam"]["fullName"] and game.get("homeTeam") == row["gameView"]["homeTeam"]["fullName"]:
                                            game["lines"] = matched[0]
                                            game["viewTypes"] = ["Spread", "MoneyLine", "Total"]
                                            game["oddsSource"] = "fixture"
            ensure_espn_odds_on_games(games)
            payload = build_dashboard_payload_from_espn_games(
                games,
                url=f"fixture:{fixture_path}",
                league=league,
            )
            payload["scheduleDate"] = date_value
            if include_odds:
                payload["sportsbooks"] = sorted(
                    {
                        line.get("sportsbook")
                        for game in games
                        for line in game.get("lines") or []
                        if line.get("sportsbook")
                    }
                )
                payload["sportsbookCount"] = len(payload["sportsbooks"])
            return payload

        page_props = load_page_props_from_file(fixture_path)
        return build_dashboard_payload_from_sbr(page_props, url=f"fixture:{fixture_path}", view_filter=view_filter)

    if source == "sbr":
        url = build_odds_url(date_value, odds_slug=league_config.sbr_odds_slug or "mlb-baseball")
        page_props = get_page_props(url, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl)
        payload = build_dashboard_payload_from_sbr(page_props, url=url, view_filter=view_filter)
        payload["scheduleDate"] = date_value
        return payload

    url = f"espn://{league_config.espn_path}/scoreboard?dates={date_value.replace('-', '')}"
    scoreboard = fetch_scoreboard(
        league,
        date_value,
        retries=retries,
        retry_delay=retry_delay,
        verify_ssl=verify_ssl,
    )
    games = parse_scoreboard(scoreboard, league=league)

    if include_enrichment:
        schedule_context = fetch_rolling_schedule_games(
            league,
            date_value,
            lookback_days=7,
            current_games=games,
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )
        enrich_games(
            games,
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )
        enrich_games_with_providers(
            games,
            league=league,
            schedule_context_games=schedule_context,
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )
        clear_rolling_schedule_cache()

    if include_odds and league_config.supports_sbr_odds:
        merge_sbr_odds_into_games(
            games,
            league=league,
            date_value=date_value,
            view_filter=view_filter,
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )

    ensure_espn_odds_on_games(games)

    payload = build_dashboard_payload_from_espn_games(games, url=url, league=league)
    payload["scheduleDate"] = date_value
    payload["scheduleTimezone"] = get_schedule_timezone(league)
    payload["defaultScheduleDate"] = default_game_date(league)

    if include_odds:
        payload["sportsbooks"] = sorted(
            {line.get("sportsbook") for game in games for line in game.get("lines") or [] if line.get("sportsbook")}
        )
        payload["sportsbookCount"] = len(payload["sportsbooks"])

    return payload


def _attach_offline_enrichment_samples(games: list[dict[str, Any]]) -> None:
    """Attach a saved ESPN summary fixture to matching games for offline reasoning demos."""
    sample_path = Path(__file__).resolve().parent / "tests" / "fixtures" / "espn_summary_401815776.json"
    if not sample_path.is_file():
        return

    with open(sample_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    for game in games:
        if str(game.get("eventId")) == "401815776":
            enrich_game(game, summary_fixture=summary)
            return
