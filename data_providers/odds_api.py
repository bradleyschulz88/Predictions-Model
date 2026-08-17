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
  * each league asks only for the markets nothing else supplies, since a credit
    is charged per market -- MLB asks for `spreads` alone at 1 credit a call
    where AFL needs all three at 3
  * the result is cached for CACHE_TTL_SECONDS, and that cache is written to
    disk, because CI starts a fresh interpreter every thirty minutes and an
    in-process dict is empty on arrival every time
  * callers only ask when they actually have unpriced games to fill
  * the quota headers the API returns are surfaced, so the budget is a number
    somebody can look at rather than a surprise

Roughly 320 credits a month against the 500 allowed: AFL around 200 at four
game days a week, MLB around 120 at four calls a day through the season.

The disk cache is the load-bearing one. Without it the six-hour TTL never
applies on CI at all, and MLB alone would spend a credit on each of ~48 builds
a day -- about 1,440 a month, the whole allowance gone inside a week.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, NamedTuple

from data_providers.utils import team_match_score
from market import decimal_to_american, is_valid_american_odds
from sbr_client import SBRClientError, get_text_with_headers

API_BASE = "https://api.the-odds-api.com/v4/sports"


class LeagueOdds(NamedTuple):
    """What to ask this league for, and what it costs.

    A credit is charged per market per region, so `markets` is the price of a
    call and not a preference. Asking for a market another source already
    supplies for free is money spent on a duplicate.
    """

    sport_key: str
    regions: str
    markets: str

    @property
    def credits_per_call(self) -> int:
        return len(self.markets.split(",")) * len(self.regions.split(","))


# Only what no free source covers, market by market rather than league by
# league. Adding a whole league at h2h,spreads,totals is three credits a call;
# adding one market to an existing one is one.
LEAGUE_ODDS: dict[str, LeagueOdds] = {
    # AFL has nothing anywhere else -- no SportsBookReview board, nothing in
    # ESPN core -- so it needs the full set from Australian books.
    "afl": LeagueOdds(sport_key="aussierules_afl", regions="au", markets="h2h,spreads,totals"),
    # MLB needs exactly one market. SportsBookReview supplies the moneyline and
    # ESPN core now prices 85 of 86 totals, but ESPN publishes the runline
    # handicap with no juice attached -- {"home": "-1.5"} where the same
    # endpoint gives WNBA {"home": "-6.5 (-112)"} -- so all 62 graded runlines
    # were unvaluable. This is the only market on the board with no free price
    # anywhere, and asking for just it costs 1 credit a call rather than 3.
    "mlb": LeagueOdds(sport_key="baseball_mlb", regions="us", markets="spreads"),
}

# Kept as a name other modules and tests already import.
SPORT_KEYS = {league: config.sport_key for league, config in LEAGUE_ODDS.items()}

# Six hours. The lines that matter are the ones near kick-off, and AFL games
# are known days ahead, so this trades a little staleness for a quota that
# lasts the season. Baseball runlines move little enough that the same window
# holds: at four calls a day it is ~120 credits a month in season.
CACHE_TTL_SECONDS = 6 * 60 * 60

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_QUOTA: dict[str, Any] = {}

# Where the cache lives between builds, and the difference between this
# provider costing ~320 credits a month and blowing the whole free tier inside
# a week.
#
# CACHE_TTL_SECONDS above is six hours. CI starts a fresh interpreter every
# thirty minutes, so an in-process dict is empty on arrival every single time
# and that TTL has never once applied there -- the docstring's "the many builds
# inside that window share a single call" was true only of a long-lived
# process. Unpersisted, MLB alone would spend one credit on each of ~48 builds
# a day, about 1,440 a month against an allowance of 500.
#
# Pointed at a file the workflow restores and saves, four calls a day is what
# actually happens. Unset outside CI, where the dict is fine.
CACHE_FILE_ENV = "ODDS_API_CACHE"


def _cache_path():
    from pathlib import Path

    raw = os.environ.get(CACHE_FILE_ENV, "").strip()
    return Path(raw) if raw else None


def load_cache() -> int:
    """Seed the cache from disk. Returns how many leagues came back.

    Entries past their TTL are dropped on the way in, so a stale file cannot
    pin yesterday's price onto a live board. A missing or malformed file is not
    an error -- the provider fetches, which is what it would have done anyway,
    and spends a credit doing it.
    """
    path = _cache_path()
    if path is None or not path.is_file():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    # Valid JSON of the wrong shape is not an error either. A bare list or a
    # null parses fine and then raises AttributeError on .get, which would take
    # the build down at start-up -- the one outcome this whole function exists
    # to avoid.
    if not isinstance(payload, dict):
        return 0
    now = time.time()
    # The balance carries even when the slates do not. A build that answers
    # entirely out of cache makes no call and reads no headers, so without this
    # it can say nothing about the budget -- and since the cache is doing its
    # job, that is most builds. Measured across 11 Aug, every one of six
    # sampled builds between 03:01Z and 23:53Z was a pure cache hit, so the
    # figure this whole mechanism exists to surface was still invisible in
    # practice even after the headers started being read.
    #
    # Deliberately not TTL-checked. A credit balance does not go stale the way
    # a price does; last month's remaining count is still the last thing the
    # API said, and `asOf` says when. Wrong-but-dated beats absent here.
    carried = payload.get("quota")
    if isinstance(carried, dict):
        _QUOTA.update({
            key: value for key, value in carried.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })

    kept = 0
    for sport_key, entry in (payload.get("entries") or {}).items():
        try:
            stamp = float(entry["fetchedAt"])
            events = entry["events"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(events, list) or now - stamp >= CACHE_TTL_SECONDS:
            continue
        _CACHE[sport_key] = (stamp, events)
        kept += 1
    return kept


def save_cache() -> int:
    """Write the cache back out. Returns how many leagues were written."""
    path = _cache_path()
    if path is None:
        return 0
    now = time.time()
    entries = {
        sport_key: {"fetchedAt": stamp, "events": events}
        for sport_key, (stamp, events) in _CACHE.items()
        if now - stamp < CACHE_TTL_SECONDS
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"entries": entries, "quota": dict(_QUOTA)}), encoding="utf-8"
        )
    except OSError:
        return 0
    return len(entries)


def is_configured() -> bool:
    """Whether a key is present. Absent is the normal case, not a failure."""
    return bool(os.environ.get("ODDS_API_KEY", "").strip())


def quota_status() -> dict[str, Any]:
    """Whatever the last call reported about the remaining monthly budget.

    Empty when nothing has been fetched this run -- a cache hit costs no credit
    and reports no headers, so an empty dict means "no call was made", not "no
    budget left".
    """
    return dict(_QUOTA)


# What the API reports about the budget, and what to call each one here. Values
# arrive as strings; anything that will not parse as an integer is dropped
# rather than published as a number that is not one.
QUOTA_HEADERS = {
    "x-requests-remaining": "remaining",
    "x-requests-used": "used",
    "x-requests-last": "lastCallCost",
}


def _record_quota(headers: dict[str, str]) -> None:
    """Pull the credit balance out of a response.

    This is the only place the budget becomes visible. Before it existed the
    module declared `_QUOTA`, exposed `quota_status()` and had the build print
    the result, but no code ever wrote to the dict -- `get_text` returns a body
    and discards the response -- so the quota line never printed once. The spend
    was invisible for as long as the provider had been live.
    """
    # Lower-cased again here rather than trusting the fetcher to have done it.
    # HTTP header names are case-insensitive, this is the only reader of them,
    # and getting it wrong costs a silent empty dict that looks exactly like
    # "no call was made" -- which is the failure this function exists to end.
    folded = {str(key).lower(): value for key, value in headers.items()}
    read_any = False
    for header, name in QUOTA_HEADERS.items():
        raw = folded.get(header)
        if raw is None:
            continue
        try:
            _QUOTA[name] = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            continue
        read_any = True
    # When the reading was taken. A balance carried in from the disk cache is
    # the last thing the API said and still worth printing, but it must not be
    # mistaken for a live figure -- most builds answer entirely from cache and
    # spend nothing, so an unstamped number would look like it was refreshed
    # every build when it was not.
    if read_any:
        _QUOTA["asOf"] = int(time.time())


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
    try:
        converted = decimal_to_american(value)
    except (OverflowError, ValueError):
        # round() on an infinity raises OverflowError, and a price field
        # carrying one should cost this outcome, not the whole fetch.
        return None
    if not is_valid_american_odds(converted):
        return None
    return int(converted)


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
    config = LEAGUE_ODDS.get(league)
    if not config or not is_configured():
        return []
    sport_key = config.sport_key

    clock = time.time() if now is None else now
    cached = _CACHE.get(sport_key)
    if cached and clock - cached[0] < CACHE_TTL_SECONDS:
        return cached[1]

    url = (
        f"{API_BASE}/{sport_key}/odds"
        f"?regions={config.regions}&markets={config.markets}&oddsFormat=decimal"
        f"&apiKey={os.environ['ODDS_API_KEY'].strip()}"
    )
    try:
        body, headers = get_text_with_headers(url, retries=2, verify_ssl=verify_ssl)
        _record_quota(headers)
        payload = json.loads(body)
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
    # Gated on the market, not just the league. MLB is configured here for its
    # runline and asks for `spreads` alone, so a call made from this function
    # could not return a moneyline even if it succeeded -- it would spend the
    # credit and attach nothing.
    config = LEAGUE_ODDS.get(league)
    if not unpriced or not config or "h2h" not in config.markets or not is_configured():
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


def fill_missing_spread_prices(
    games: list[dict[str, Any]],
    *,
    league: str,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Price the spread on games that already have a moneyline.

    A separate entry point from attach_odds_to_games on purpose. That one asks
    only about games with no moneyline at all, which is right for AFL and
    exactly wrong here: every MLB game has a moneyline from SportsBookReview,
    so baseball would never be looked at and no credit would ever be spent.
    That same gate is why ESPN core was never asked about MLB side markets in
    the first place.

    Only the Spread rows are merged. The moneyline and total already have free
    sources, and a second copy of either would land in the consensus twice.
    """
    # Imported here rather than at module scope: espn_odds imports
    # mlb_predictions, which imports this module, and a top-level import would
    # close that cycle.
    from espn_odds import has_priced_market

    stats: dict[str, Any] = {
        "league": league, "considered": 0, "priced": 0, "books": [],
        "configured": is_configured(),
    }
    config = LEAGUE_ODDS.get(league)
    if not config or "spreads" not in config.markets or not is_configured():
        return stats

    wanted = [
        game for game in games
        if not has_priced_market(game.get("lines") or [], "Spread")
    ]
    stats["considered"] = len(wanted)
    if not wanted:
        return stats

    events = fetch_league_odds(league, verify_ssl=verify_ssl)
    if not events:
        return stats

    for game in wanted:
        event = _match_event(game, events)
        if not event:
            continue
        lines = [
            line for line in lines_from_event(event)
            if "Spread" in (line.get("viewType") or "")
        ]
        if not lines:
            continue
        game.setdefault("lines", [])
        game["lines"].extend(lines)
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

    That is not enough on its own, because a shared city scores high both ways
    round. "New York Yankees" against "New York Mets" scores 0.667 on two of
    three matching words, so an event with the two clubs the right way round
    and one with them swapped both clear a 0.6 bar -- and a swapped match
    hands the home side the away side's handicap and price, turning a -1.5
    favourite into a +1.5 underdog. Silently, on a market whose whole purpose
    is to be valued.

    So orientation has to be decided rather than assumed: an event only counts
    if it fits this fixture better than its mirror image does. Identical clubs
    score 1.0 forwards and 0.667 reversed, which is a clear answer, while the
    swapped event scores 0.667 forwards and 1.0 reversed and is rejected. A
    per-side floor would not catch this, because 0.667 clears any floor low
    enough to tolerate the naming differences this matcher exists for.
    """
    home = str(game.get("homeTeam") or "")
    away = str(game.get("awayTeam") or "")
    best, best_score = None, 0.0
    for event in events:
        event_home = str(event.get("home_team") or "")
        event_away = str(event.get("away_team") or "")
        score = (
            team_match_score(event_home, home) + team_match_score(event_away, away)
        ) / 2.0
        mirrored = (
            team_match_score(event_home, away) + team_match_score(event_away, home)
        ) / 2.0
        # Ties count as ambiguous and are dropped: if a fixture reads the same
        # both ways there is nothing to tell the sides apart, and guessing is
        # how the wrong price gets attached.
        if score <= mirrored:
            continue
        if score > best_score:
            best, best_score = event, score
    return best if best_score >= 0.6 else None
