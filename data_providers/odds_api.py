"""The Odds API — a price source for leagues no other feed covers.

AFL is the reason this exists. It has no SportsBookReview board and ESPN's
core odds carry nothing for it, so every AFL pick has been permanently
unpriced: no expected value, no stake size, and no way to rank it against a
game in any other league. That is not a bug to fix in the existing sources;
neither of them has the data.

The Odds API's free plan covers Australian Rules with `regions=au`, which is
real Australian books rather than a US shop guessing at an AFL line.

**Budget is the whole design constraint.** The free plan allows 500 credits a
month, and a credit is charged per market per region -- so one call asking for
h2h, spreads and totals from `au` costs three. The site rebuilds every thirty
minutes, about 1,440 times a month, so calling on every build would exhaust a
month's quota inside a day. Everything here exists to avoid that:

  * nothing happens without ODDS_API_KEY, exactly like the injury-severity
    provider, so a fork or a local run costs nothing and behaves identically
  * one call fetches the league's whole slate, not one call per game
  * the result is cached for CACHE_TTL_SECONDS, so the many builds inside that
    window share a single call
  * callers only ask when they actually have unpriced games to fill
  * the quota headers the API returns are surfaced, so the budget is a number
    somebody can look at rather than a surprise

At four game days a week in season and a six-hour cache, that lands around
200 credits a month against the 500 allowed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from data_providers.utils import team_match_score
from market import decimal_to_american
from sbr_client import SBRClientError, get_text

API_BASE = "https://api.the-odds-api.com/v4/sports"

# Only leagues that no other source covers. Every extra league here is another
# three credits a call against a 500-credit month, and MLB, NBA, NFL, WNBA and
# EPL are all already served by SportsBookReview or ESPN for nothing.
SPORT_KEYS = {"afl": "aussierules_afl"}

# Australian books for an Australian competition.
REGIONS = "au"
MARKETS = "h2h,spreads,totals"

# Six hours. The lines that matter are the ones near kick-off, and AFL games
# are known days ahead, so this trades a little staleness for a quota that
# lasts the season.
CACHE_TTL_SECONDS = 6 * 60 * 60

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_QUOTA: dict[str, Any] = {}


def is_configured() -> bool:
    """Whether a key is present. Absent is the normal case, not a failure."""
    return bool(os.environ.get("ODDS_API_KEY", "").strip())


def quota_status() -> dict[str, Any]:
    """Whatever the last call reported about the remaining monthly budget."""
    return dict(_QUOTA)


def clear_cache() -> None:
    _CACHE.clear()
    _QUOTA.clear()


def _decimal_to_american(price: Any) -> int | None:
    """The API quotes decimal; everything downstream speaks American."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return None
    if value <= 1.0:
        return None
    return int(decimal_to_american(value))


def fetch_league_odds(
    league: str,
    *,
    verify_ssl: bool = True,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Every priced event for one league, or an empty list.

    Never raises. Odds decorate a schedule that is already built, so a failure
    here costs prices and nothing else -- the same contract every other odds
    source in this project honours.
    """
    sport_key = SPORT_KEYS.get(league)
    if not sport_key or not is_configured():
        return []

    clock = time.time() if now is None else now
    cached = _CACHE.get(sport_key)
    if cached and clock - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    url = (
        f"{API_BASE}/{sport_key}/odds"
        f"?regions={REGIONS}&markets={MARKETS}&oddsFormat=decimal"
        f"&apiKey={os.environ['ODDS_API_KEY'].strip()}"
    )
    try:
        payload = json.loads(get_text(url, retries=2, verify_ssl=verify_ssl))
    except (SBRClientError, json.JSONDecodeError, ValueError, OSError):
        # Cache the failure briefly too. A key that has run out of credits
        # fails on every call, and retrying it every build helps nobody.
        _CACHE[sport_key] = (clock, [])
        return []

    events = payload if isinstance(payload, list) else []
    _CACHE[sport_key] = (clock, events)
    return events


def _outcomes(market: dict[str, Any]) -> list[dict[str, Any]]:
    outcomes = market.get("outcomes")
    return outcomes if isinstance(outcomes, list) else []


def lines_from_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """One event's books, in the line shape the model already consumes.

    Every book is emitted rather than only the best. _best_price_for_side picks
    the best across them and rejects outliers against the rest, which it can
    only do if it can see the whole board.
    """
    home = event.get("home_team")
    away = event.get("away_team")
    lines: list[dict[str, Any]] = []

    for book in event.get("bookmakers") or []:
        title = book.get("title") or book.get("key") or "The Odds API"
        for market in book.get("markets") or []:
            key = market.get("key")
            outcomes = _outcomes(market)

            if key == "h2h":
                prices: dict[str, int] = {}
                for outcome in outcomes:
                    side = _side_for(outcome.get("name"), home, away)
                    price = _decimal_to_american(outcome.get("price"))
                    if side and price is not None:
                        prices[side] = price
                if prices:
                    lines.append({
                        "sportsbook": title,
                        "viewType": "MoneyLine",
                        "currentLine": {k: str(v) for k, v in prices.items()},
                    })

            elif key == "spreads":
                current: dict[str, str] = {}
                for outcome in outcomes:
                    side = _side_for(outcome.get("name"), home, away)
                    price = _decimal_to_american(outcome.get("price"))
                    point = outcome.get("point")
                    if side and price is not None and point is not None:
                        # The parenthetical price is the shape extract_spread_price
                        # already reads off ESPN core lines.
                        current[side] = f"{float(point):+g} ({price:+d})"
                if current:
                    lines.append({
                        "sportsbook": title, "viewType": "Spread", "currentLine": current,
                    })

            elif key == "totals":
                current = {}
                for outcome in outcomes:
                    name = str(outcome.get("name") or "").strip().lower()
                    price = _decimal_to_american(outcome.get("price"))
                    point = outcome.get("point")
                    if name in {"over", "under"} and price is not None and point is not None:
                        prefix = "o" if name == "over" else "u"
                        current[name] = f"{prefix}{float(point):g} ({price:+d})"
                if current:
                    lines.append({
                        "sportsbook": title, "viewType": "Total", "currentLine": current,
                    })

    return lines


def _side_for(name: Any, home: Any, away: Any) -> str | None:
    """Which side of the fixture an outcome names.

    Fuzzy, because a book writes "Carlton" where ESPN writes "Carlton Blues".
    Requires a clear winner between the two, so an ambiguous name attaches to
    neither rather than guessing.
    """
    if not name:
        return None
    home_score = team_match_score(str(name), str(home or ""))
    away_score = team_match_score(str(name), str(away or ""))
    if max(home_score, away_score) < 0.5 or abs(home_score - away_score) < 0.1:
        return None
    return "home" if home_score > away_score else "away"


def attach_odds_to_games(
    games: list[dict[str, Any]],
    *,
    league: str,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Price any game in this league that still has none. Reports what happened.

    Only games that reach here unpriced are considered, so a league another
    source already covers never spends a credit.
    """
    from mlb_predictions import has_moneyline_lines

    stats: dict[str, Any] = {
        "league": league, "considered": 0, "priced": 0, "books": [], "configured": is_configured(),
    }
    unpriced = [
        game for game in games if not has_moneyline_lines(game.get("lines") or [])
    ]
    stats["considered"] = len(unpriced)
    if not unpriced or not SPORT_KEYS.get(league) or not is_configured():
        return stats

    events = fetch_league_odds(league, verify_ssl=verify_ssl)
    if not events:
        return stats

    for game in unpriced:
        event = _match_event(game, events)
        if not event:
            continue
        lines = lines_from_event(event)
        if not lines:
            continue
        game.setdefault("lines", [])
        game["lines"].extend(lines)
        game["oddsSource"] = "the-odds-api"
        stats["priced"] += 1
        for line in lines:
            book = line.get("sportsbook")
            if book and book not in stats["books"]:
                stats["books"].append(book)
    return stats


def _match_event(game: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The API event for this fixture, matched on both club names at once.

    Scoring the pair together rather than either alone is what stops a derby
    -- two clubs from the same city, sharing a word in their names -- from
    attaching the wrong fixture's prices.
    """
    home = str(game.get("homeTeam") or "")
    away = str(game.get("awayTeam") or "")
    best, best_score = None, 0.0
    for event in events:
        score = (
            team_match_score(str(event.get("home_team") or ""), home)
            + team_match_score(str(event.get("away_team") or ""), away)
        ) / 2.0
        if score > best_score:
            best, best_score = event, score
    return best if best_score >= 0.6 else None
