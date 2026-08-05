"""Transform schedule and odds data into dashboard payloads."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_providers.utils import team_match_score
from espn_client import ESPNClientError, fetch_scoreboard, parse_scoreboard
from espn_odds import fill_missing_moneylines, fill_missing_side_market_prices
from espn_enrichment import enrich_game, enrich_games, ensure_espn_odds_on_games
from data_providers import enrich_games_with_providers
from mlb_predictions import apply_predictions, has_moneyline_lines, is_publishable_pick
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
) -> dict[str, Any]:
    """Attach SBR lines to games. Returns what happened, for diagnosis.

    The returned stats distinguish the two ways this silently produces nothing:
    `fetched=False` means the page or its odds table never arrived (a bad slug,
    a league SBR does not cover, or an outage), while `fetched=True` with
    `rows > 0` and `matched == 0` means the odds arrived fine and the team
    names did not line up. Those need opposite fixes and used to look identical
    from outside, because the guard swallows both.
    """
    stats: dict[str, Any] = {
        "league": league,
        "date": date_value,
        "configured": True,
        "fetched": False,
        "rows": 0,
        "games": len(games),
        "matched": 0,
        # Matched means the game was found on SBR; priced means it came back
        # with a moneyline. WNBA matched fine and was never priced.
        "priced": 0,
        "viewTypes": [],
        "unmatched": [],
        "sbrNames": [],
        # Enrichment has already run by this point, so this says whether the
        # independent ESPN fallback has anything either. When SBR and ESPN both
        # come up empty for a league, the cause is upstream of team matching.
        "espnOddsGames": sum(
            1 for game in games if (game.get("enrichment") or {}).get("espnOdds")
        ),
    }

    league_config = get_league(league)
    odds_slug = league_config.sbr_odds_slug
    if not odds_slug:
        stats["configured"] = False
        return stats

    # Odds are a garnish on the schedule, never a precondition for it. Parsing
    # has to sit inside the guard alongside fetching: SBR serves a valid page
    # with no `oddsTables` whenever a league has no priced slate, and
    # get_game_rows raises on that. Escaping here aborted the whole payload and
    # threw away a schedule that had already been fetched and parsed -- which is
    # why NFL/NBA/WNBA/EPL published zero games on days SBR had no board, while
    # MLB never noticed because SBR always prices MLB.
    odds_by_matchup: dict[str, list[dict[str, Any]]] = {}
    view_types_by_matchup: dict[str, list[str]] = {}

    try:
        page_props = get_page_props(
            build_odds_url(date_value, odds_slug=odds_slug),
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
        )
        rows = get_game_rows(page_props)
    except SBRClientError:
        return stats

    stats["fetched"] = True
    stats["rows"] = len(rows)

    for row in rows:
        summary = game_summary(row)
        key = matchup_key(summary.get("awayTeam"), summary.get("homeTeam"))
        odds_by_matchup[key] = collect_odds_lines(row, view_filter=view_filter)
        view_types_by_matchup[key] = collect_view_types(row)
        stats["sbrNames"].append(f"{summary.get('awayTeam')} @ {summary.get('homeTeam')}")

    for game in games:
        matched = _find_sbr_odds_match(
            game.get("awayTeam"),
            game.get("homeTeam"),
            odds_by_matchup,
            view_types_by_matchup,
        )
        if matched:
            _attach_sbr_lines(game, matched[0], matched[1])
            stats["matched"] += 1
            # Matching is not the same as pricing. SBR can return a row whose
            # only markets are spread and total, which attaches lines and still
            # leaves the game unpriced -- the model needs a moneyline. Counting
            # matches alone reported that as success.
            if has_moneyline_lines(game.get("lines") or []):
                stats["priced"] += 1
            else:
                for line in game.get("lines") or []:
                    view = line.get("viewType")
                    if view and view not in stats["viewTypes"]:
                        stats["viewTypes"].append(view)
        else:
            stats["unmatched"].append(f"{game.get('awayTeam')} @ {game.get('homeTeam')}")

    return stats


def _report_core_odds(stats: dict[str, Any], date_value: str) -> None:
    """Report only when the last-resort source was actually needed.

    Silence means SBR and the summary already priced everything, which is the
    normal MLB case. A line here means a league that had no market before now
    has one -- or that this source could not help either.
    """
    considered = stats.get("considered") or 0
    if not considered:
        return

    where = f"{stats['league']} {date_value}"
    if stats.get("priced"):
        books = ", ".join(stats["books"][:3]) or "unknown"
        print(
            f"Odds: {where}: ESPN core priced {stats['priced']}/{considered} "
            f"previously unpriced games via {books}",
            flush=True,
        )
    else:
        print(
            f"Odds: {where}: ESPN core had no moneyline for any of "
            f"{considered} unpriced games either",
            flush=True,
        )


def _report_odds_api(stats: dict[str, Any], date_value: str) -> None:
    """Only speak when this source was actually reached for.

    Silence covers both the normal case -- every other feed already priced the
    slate -- and the unconfigured one, which is not a failure: no key means no
    metered calls, and the build behaves exactly as it did before.
    """
    considered = stats.get("considered") or 0
    if not considered or not stats.get("configured"):
        return
    where = f"{stats['league']} {date_value}"
    if stats.get("priced"):
        books = ", ".join(stats["books"][:3]) or "unknown"
        print(
            f"Odds: {where}: The Odds API priced {stats['priced']}/{considered} "
            f"previously unpriced games via {books}",
            flush=True,
        )
    else:
        print(
            f"Odds: {where}: The Odds API returned nothing for {considered} unpriced games",
            flush=True,
        )


def _price_source_coverage(
    league_config: Any, core_stats: dict[str, Any]
) -> dict[str, Any] | None:
    """Why this league has the prices it has -- or does not have any.

    "No prices" has two causes that look identical on the board: a feed that
    failed today, and a league no feed has ever covered. AFL is the second
    kind -- no SportsBookReview board, and ESPN's core odds carry nothing for
    it -- so every one of its picks is unpriced, permanently, and no amount of
    waiting will change that.

    Until now the only record of it was a line on stdout that scrolls away in
    Actions and a comment in espn_odds.py. Recording it per build turns a claim
    into a measurement: if ESPN ever does start covering a league, this flips
    on its own, and nobody has to take the comment's word for it.
    """
    considered = core_stats.get("considered") or 0
    if not (getattr(league_config, "supports_sbr_odds", False) or considered):
        return None
    return {
        "sbrBoard": bool(getattr(league_config, "supports_sbr_odds", False)),
        # Games that reached the last-resort source still lacking any price.
        "espnCoreTried": considered,
        "espnCorePriced": core_stats.get("priced") or 0,
        # True only when a league with no SBR board asked ESPN core and got
        # nothing back for anything -- the signature of "no source exists".
        "noSourceFound": bool(
            not getattr(league_config, "supports_sbr_odds", False)
            and considered
            and not (core_stats.get("priced") or 0)
        ),
    }


def _report_odds_merge(stats: dict[str, Any]) -> None:
    """Say something only when a league that should have prices did not get them.

    A working league stays silent so the log does not fill with noise; the point
    is to name the failure mode, since "no prices" has two very different causes
    that look the same from outside.
    """
    if not stats.get("configured") or not stats.get("games"):
        return
    # Priced is the bar, not matched: a row carrying only spreads and totals
    # attaches lines and still leaves the game unpriced.
    if stats.get("priced"):
        return

    where = f"{stats['league']} {stats['date']}"
    espn = stats.get("espnOddsGames", 0)
    espn_note = (
        f" (ESPN has odds on {espn}/{stats['games']}, so the fallback should cover it)"
        if espn
        else f" (ESPN has no odds either, on any of {stats['games']})"
    )

    if stats.get("matched"):
        markets = ", ".join(stats["viewTypes"]) or "none"
        print(
            f"Odds: {where}: SBR matched {stats['matched']}/{stats['games']} games but "
            f"none carry a moneyline -- markets returned: {markets}{espn_note}",
            flush=True,
        )
        return

    if not stats.get("fetched"):
        print(
            f"Odds: {where}: SBR returned no odds table -- check the slug "
            f"({get_league(stats['league']).sbr_odds_slug!r}) or whether SBR covers "
            f"this league{espn_note}",
            flush=True,
        )
        return

    if not stats.get("rows"):
        print(f"Odds: {where}: SBR page listed no games{espn_note}", flush=True)
        return

    # The informative case: prices arrived, names did not line up. Printing both
    # sides is what turns this from a mystery into a one-line mapping fix.
    print(
        f"Odds: {where}: SBR listed {stats['rows']} games but matched 0/{stats['games']} "
        f"-- team names disagree",
        flush=True,
    )
    print(f"  ours: {'; '.join(stats['unmatched'][:4])}", flush=True)
    print(f"  SBR : {'; '.join(stats['sbrNames'][:4])}", flush=True)


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
        publishable = [
            game
            for game in games
            if is_publishable_pick(game.get("prediction"), game.get("league"))
        ]
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

    # Once the scoreboard has parsed, the schedule is the product and every step
    # below only decorates it. A provider that is down must cost us its own
    # contribution and nothing else -- an exception escaping any of these used to
    # discard the whole slate and publish "0 games" for a league that had a full
    # card. Failures are recorded on the payload so a silent degradation still
    # shows up in the coverage report rather than looking like an empty day.
    degraded: list[str] = []

    def _optional(step: str, run) -> None:
        try:
            run()
        except Exception as exc:  # noqa: BLE001 - deliberately broad; see above
            print(f"Warning: {league} {date_value}: {step} unavailable: {exc}", flush=True)
            degraded.append(step)

    if include_enrichment:
        schedule_context: list[dict[str, Any]] = []

        def _schedule_context() -> None:
            nonlocal schedule_context
            schedule_context = fetch_rolling_schedule_games(
                league,
                date_value,
                lookback_days=7,
                current_games=games,
                retries=retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            )

        _optional("schedule context", _schedule_context)
        _optional(
            "ESPN enrichment",
            lambda: enrich_games(games, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl),
        )
        _optional(
            "provider enrichment",
            lambda: enrich_games_with_providers(
                games,
                league=league,
                schedule_context_games=schedule_context,
                retries=retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            ),
        )
        clear_rolling_schedule_cache()

    if include_odds and league_config.supports_sbr_odds:
        _optional(
            "SBR odds",
            lambda: _report_odds_merge(
                merge_sbr_odds_into_games(
                    games,
                    league=league,
                    date_value=date_value,
                    view_filter=view_filter,
                    retries=retries,
                    retry_delay=retry_delay,
                    verify_ssl=verify_ssl,
                )
            ),
        )

    _optional("ESPN odds", lambda: ensure_espn_odds_on_games(games))

    # Last resort, and only for games still without a moneyline. SBR prices
    # most of MLB and the ESPN summary catches some of the rest, so on a normal
    # day this makes no requests at all. It exists for WNBA, which SBR returns
    # with spread and total but never a moneyline, and for AFL, which has no
    # SBR board at all -- between them about a third of the graded history has
    # never had a market to anchor to.
    core_stats: dict[str, Any] = {}

    def _core_odds() -> None:
        core_stats.update(
            fill_missing_moneylines(
                games,
                league=league,
                retries=retries,
                retry_delay=retry_delay,
                verify_ssl=verify_ssl,
            )
        )
        _report_core_odds(core_stats, date_value)

    if include_odds:
        _optional("ESPN core odds", _core_odds)

    # Second core pass, for games that already have a moneyline but whose
    # total and spread carry no price. SBR posts those lines without odds, so
    # an MLB game looked fully covered while being impossible to value on
    # anything but the moneyline -- 11 priced spreads out of 58, the rest
    # runlines that could not be valued at all.
    def _core_side_markets() -> None:
        stats = fill_missing_side_market_prices(games, league=league, verify_ssl=verify_ssl)
        if stats.get("priced"):
            print(
                f"Odds: {league} {date_value}: ESPN core priced side markets on "
                f"{stats['priced']}/{stats['considered']} games "
                f"({stats['fetched']} fetched, {stats['cached']} cached)",
                flush=True,
            )

    # ON by default. It was off, on a diagnosis that turned out to be wrong.
    #
    # The reasoning was that this pass took the schedule feeds down: it fires
    # one ESPN core request per unpriced game, the first build after it began
    # firing had all six feeds fail at once, and that looks like an IP-level
    # rate limit. It was not. The feeds were failing on HTTP 403 from
    # site.api.espn.com, which rejects any request claiming to be a browser
    # without a browser's TLS fingerprint; an honest project User-Agent gets
    # 200, and has since. The timings never fit the rate-limit story either --
    # the 11:00Z artifact was healthy at 717,559 bytes and the 23:30Z one was
    # empty, while this pass only reached main at 00:13Z the next day.
    #
    # This pass talks to sports.core.api.espn.com, a different host, which
    # served 200 OK on every header profile probed on 2026-08-05 including the
    # bare one. It was never the host that broke.
    #
    # The volume concern was real and is now addressed rather than avoided: the
    # cache persists across builds via ESPN_SIDE_MARKET_CACHE, so the rate is
    # the intended one fetch per game per hour instead of one per game per
    # build. Set ESPN_SIDE_MARKET_ODDS=0 to switch it back off.
    #
    # What it buys, measured on the 2026-08-05 22:46Z build: totals went from
    # 71/81 priced to 85/86. Runlines did not move, and will not from here.
    #
    # That was the goal and it is not reachable through this source. ESPN core
    # returns a spread row for baseball, it survives the "Spread" filter, and
    # extract_spread_price correctly reads nothing out of it, because the row
    # carries no juice:
    #
    #     mlb   Spread  {"home": "-1.5",        "away": "+1.5"}
    #     wnba  Spread  {"home": "-6.5 (-112)", "away": "+6.5 (-108)"}
    #
    # DraftKings publishes the runline handicap here without a price. Nothing
    # is broken; the number does not exist on this endpoint. It explains the
    # graded record exactly -- all 13 priced spreads were WNBA.
    #
    # Pricing runlines needs a different source, and now uses one: the Odds API
    # pass below asks baseball for the `spreads` market only. See
    # scripts/diagnose_side_markets.py to re-check whether ESPN ever starts
    # publishing the price, at which point the paid call can be dropped.
    if include_odds and os.environ.get("ESPN_SIDE_MARKET_ODDS", "1").strip() not in {"0", "false", "no"}:
        _optional("ESPN core side markets", _core_side_markets)

    # Truly last: a paid-tier-free API that covers leagues nothing else does.
    # Costs a metered credit, so it runs only after every free source has had
    # its turn and only if games are still unpriced, and does nothing at all
    # without a key. AFL is the case it exists for -- no SBR board, nothing in
    # ESPN core, so its picks have never had a price to compute value from.
    api_stats: dict[str, Any] = {}

    def _odds_api() -> None:
        from data_providers.odds_api import attach_odds_to_games

        api_stats.update(
            attach_odds_to_games(games, league=league, verify_ssl=verify_ssl)
        )
        _report_odds_api(api_stats, date_value)

    # Second Odds API pass, for a market rather than a whole game. A league can
    # be fully priced on the moneyline and still have no spread price anywhere
    # free -- which is exactly MLB, where ESPN publishes the runline handicap
    # with no juice on it. attach_odds_to_games above only looks at games with
    # no moneyline at all, so baseball never reaches it and no credit is spent.
    #
    # Shares the same six-hour cache as the pass above, so on a day this has
    # already fetched, it costs nothing further.
    def _odds_api_spreads() -> None:
        from data_providers.odds_api import fill_missing_spread_prices

        stats = fill_missing_spread_prices(games, league=league, verify_ssl=verify_ssl)
        if stats.get("priced"):
            print(
                f"Odds: {league} {date_value}: The Odds API priced spreads on "
                f"{stats['priced']}/{stats['considered']} games "
                f"({', '.join(stats['books'][:3])})",
                flush=True,
            )

    if include_odds:
        _optional("The Odds API", _odds_api)
        _optional("The Odds API spreads", _odds_api_spreads)

    payload = build_dashboard_payload_from_espn_games(games, url=url, league=league)
    if degraded:
        payload["degraded"] = degraded
    coverage = _price_source_coverage(league_config, core_stats)
    if coverage:
        payload["priceCoverage"] = coverage
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
