"""Moneylines from ESPN's core API, for games no other source prices.

Why this exists
---------------
WNBA has never had a price on any of its 115 logged picks, and AFL has never
had an odds source at all. Between them that is roughly a third of the model's
graded history scored without a market to anchor to, so their EV and ROI read
"not measurable" rather than a number.

SportsBookReview is the existing source and it does not solve either. For WNBA
it returns the games but only spread and total markets, never a moneyline --
which the model cannot use, since `marketLogit` is built from moneyline alone.
For AFL it has no board at all.

ESPN's core API carries moneylines for WNBA. Measured on the live build of
2026-07-29: 8 WNBA games priced via DraftKings that had never had a price.
It does NOT carry them for AFL -- every AFL game came back empty -- so AFL
stays unpriced and the circuit breaker below stops us asking forever.

    https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}
        /events/{event}/competitions/{event}/odds

It needs no key, has no published quota, and -- the part that matters most --
it is the same origin as the schedule. Team names and event ids are identical
by construction, so none of the fuzzy team matching that SBR needs applies
here. There is nothing to mismatch.

Cost control
------------
This is a per-event endpoint, so it is called ONLY for games still lacking a
moneyline after SBR and the ESPN summary have both had a go. On a normal day
that is zero calls for MLB and a handful for WNBA. A league ESPN does not
cover gives up after a few consecutive empties rather than retrying every
game on every build.

Shape
-----
`items[]`, one per sportsbook, each with `homeTeamOdds.moneyLine` and
`awayTeamOdds.moneyLine` as American prices, `drawOdds.moneyLine` for soccer,
and an `open` block carrying the same fields at open. Providers that are
prediction sites rather than sportsbooks are skipped -- their numbers are
model output, and folding a model's own guess back in as "the market" would be
circular.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from espn_client import ESPN_USER_AGENT
from market import is_valid_american_odds
from sbr_client import SBRClientError, get_text
from shared_utils import write_json
from sports_config import LeagueConfig, get_league

_PRICE_IN_PARENS = __import__('re').compile(r'\(([+-]\d+)\)')

ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports"

# These appear alongside real books in the same items[] list but are model or
# consensus feeds, not markets. Treating a prediction site as the market would
# make the model's own anchor circular.
NON_BOOK_PROVIDERS = {
    "consensus",
    "teamrankings",
    "numberfire",
    "espn analytics",
    "accuscore",
}

# Provider 2000 is a legacy three-way decimal feed whose numbers do not agree
# with the American prices in the same payload.
SKIP_PROVIDER_IDS = {"2000"}


class ESPNOddsError(SBRClientError):
    """Error while fetching or parsing ESPN core-API odds."""


def build_odds_url(league: LeagueConfig | str, event_id: str | int) -> str:
    """Core-API odds URL for one event.

    `espn_path` is already "{sport}/{league}" (e.g. "basketball/wnba"), which is
    exactly the split the core API wants, so the same config drives both the
    scoreboard and this.
    """
    league_config = get_league(league) if isinstance(league, str) else league
    sport, _, league_slug = league_config.espn_path.partition("/")
    return (
        f"{ESPN_CORE_BASE}/{sport}/leagues/{league_slug}"
        f"/events/{event_id}/competitions/{event_id}/odds"
    )


def _american(value: Any) -> int | None:
    """American odds from ESPN's several spellings. None when unusable.

    ESPN gives integers on some feeds and strings like "+120" or "EVEN" on
    others, and uses 0 as a null.

    Anything outside the valid American range is rejected, not just 0. The band
    between -100 and +100 is not a price at all (see
    `market.is_valid_american_odds`), and a value from it does not merely look
    odd downstream -- it wins the best-price shop and publishes a fake edge.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if is_valid_american_odds(value) else None
    text = str(value).strip().replace("+", "")
    if not text:
        return None
    if text.upper() in {"EVEN", "EV", "PK"}:
        return 100
    try:
        parsed = float(text)
    except ValueError:
        return None
    return int(parsed) if is_valid_american_odds(parsed) else None


def _side_moneyline(block: Any) -> int | None:
    """Moneyline for one side, across the shapes ESPN uses.

    Flat `moneyLine` is the common one; `current.moneyLine.american` and
    `close.odds` show up on other feeds for the same competition.
    """
    if not isinstance(block, dict):
        return None

    direct = _american(block.get("moneyLine"))
    if direct is not None:
        return direct

    for key in ("current", "close", "open"):
        nested = block.get(key)
        if not isinstance(nested, dict):
            continue
        moneyline = nested.get("moneyLine")
        if isinstance(moneyline, dict):
            found = _american(moneyline.get("american") or moneyline.get("value"))
            if found is not None:
                return found
        found = _american(moneyline if not isinstance(moneyline, dict) else None)
        if found is not None:
            return found
        found = _american(nested.get("odds"))
        if found is not None:
            return found
    return None


def _is_book(item: dict[str, Any]) -> bool:
    provider = item.get("provider") or {}
    name = str(provider.get("name") or "").strip()
    if not name:
        # An item with no named provider cannot be attributed to a book.
        return False
    if str(provider.get("id") or "") in SKIP_PROVIDER_IDS:
        return False
    return name.casefold() not in NON_BOOK_PROVIDERS


def parse_core_odds(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Core-API odds payload -> lines in the shape the model already consumes.

    One MoneyLine line per book, so several books produce a real consensus and
    the de-vig has something to average, exactly as with SBR.
    """
    if not isinstance(payload, dict):
        return []

    lines: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict) or not _is_book(item):
            continue

        home = _side_moneyline(item.get("homeTeamOdds"))
        away = _side_moneyline(item.get("awayTeamOdds"))
        if home is None or away is None:
            # Spread- or total-only rows land here. That is the exact WNBA
            # symptom on SBR, and skipping them keeps a priceless row from
            # masquerading as a price.
            continue

        current: dict[str, Any] = {"home": home, "away": away}
        draw = _side_moneyline(item.get("drawOdds"))
        if draw is not None:
            current["draw"] = draw

        opening: dict[str, Any] = {}
        open_block = item.get("open")
        if isinstance(open_block, dict):
            open_home = _side_moneyline(open_block.get("homeTeamOdds"))
            open_away = _side_moneyline(open_block.get("awayTeamOdds"))
            if open_home is not None and open_away is not None:
                opening = {"home": open_home, "away": open_away}
                open_draw = _side_moneyline(open_block.get("drawOdds"))
                if open_draw is not None:
                    opening["draw"] = open_draw

        lines.append(
            {
                "sportsbook": str((item.get("provider") or {}).get("name")),
                "viewType": "MoneyLine",
                "currentLine": current,
                "openingLine": opening or None,
            }
        )

    return lines + _market_lines(payload)


def _number(value: Any) -> float | None:
    """A spread or total as a float. None when absent or unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip().replace("+", ""))
    except (TypeError, ValueError):
        return None


def _market_lines(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Total and Spread rows, in the shape the model already consumes.

    Split out from the moneyline pass because these rows survive a different
    test: a book can post a total with no moneyline, and that total is still
    perfectly good. Requiring a moneyline first -- which the loop above does,
    correctly, for its own purposes -- would throw those away.

    This is why the totals and spread sections never appeared on a card for a
    game ESPN priced: the parser only ever emitted MoneyLine, so
    `extract_total_line` had nothing to find and `predict_total` returned None.
    """
    lines: list[dict[str, Any]] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict) or not _is_book(item):
            continue
        book = str((item.get("provider") or {}).get("name"))

        total = _number(item.get("overUnder"))
        if total is not None:
            over_price = _american(item.get("overOdds"))
            under_price = _american(item.get("underOdds"))
            lines.append(
                {
                    "sportsbook": book,
                    "viewType": "Total",
                    # extract_total_line parses the "o8.5 (-110)" spelling SBR
                    # uses, so emit that rather than teaching it a second shape.
                    "currentLine": {
                        "over": f"o{total:g}" + (f" ({over_price:+d})" if over_price else ""),
                        "under": f"u{total:g}" + (f" ({under_price:+d})" if under_price else ""),
                    },
                    "openingLine": None,
                }
            )

        spread = _number(item.get("spread"))
        if spread is not None:
            home_price = _american((item.get("homeTeamOdds") or {}).get("spreadOdds"))
            away_price = _american((item.get("awayTeamOdds") or {}).get("spreadOdds"))
            # ESPN quotes `spread` from the home side, so the away number is its
            # mirror. Getting this backwards is a silent sign flip.
            lines.append(
                {
                    "sportsbook": book,
                    "viewType": "Spread",
                    "currentLine": {
                        "home": f"{spread:+g}" + (f" ({home_price:+d})" if home_price else ""),
                        "away": f"{-spread:+g}" + (f" ({away_price:+d})" if away_price else ""),
                    },
                    "openingLine": None,
                }
            )

    return lines


def fetch_event_odds(
    league: LeagueConfig | str,
    event_id: str | int,
    *,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
) -> list[dict[str, Any]]:
    """Moneyline lines for one event. Empty list on any failure.

    Never raises: odds decorate a schedule that is already built, so a failure
    here must cost this game its price and nothing more.
    """
    try:
        text = get_text(
            build_odds_url(league, event_id),
            retries=retries,
            retry_delay=retry_delay,
            verify_ssl=verify_ssl,
            user_agent=ESPN_USER_AGENT,
        )
        payload = json.loads(text)
    except (SBRClientError, json.JSONDecodeError, ValueError, OSError):
        return []
    return parse_core_odds(payload)


# Stop asking a league that has just told us it has nothing. ESPN carries
# moneylines for WNBA but not for AFL, and without this every AFL game is
# retried on every build forever -- roughly 400 pointless requests a day.
# Per call, not cached across builds, so the moment ESPN starts covering a
# league it is picked up on the next run with no code change.
CONSECUTIVE_EMPTY_BEFORE_GIVING_UP = 3


def fill_missing_moneylines(
    games: list[dict[str, Any]],
    *,
    league: str,
    max_events: int = 40,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    """Price any game that still has no moneyline. Returns what happened.

    Deliberately last in the chain: SBR and the ESPN summary get first refusal,
    so on a normal MLB day this makes no requests at all. The cap is a seatbelt
    against an unexpectedly huge slate, not an expected limit.

    Gives up on a league after a few consecutive empties: a slate where nothing
    is priced yet is the normal case for games days out, and hammering the
    endpoint for each one buys nothing.
    """
    from mlb_predictions import has_moneyline_lines

    stats: dict[str, Any] = {
        "league": league,
        "considered": 0,
        "fetched": 0,
        "priced": 0,
        "books": [],
        "gaveUp": False,
    }
    consecutive_empty = 0

    for game in games:
        if has_moneyline_lines(game.get("lines") or []):
            continue
        event_id = game.get("eventId")
        if not event_id:
            continue

        stats["considered"] += 1
        if stats["fetched"] >= max_events or stats["gaveUp"]:
            continue

        stats["fetched"] += 1
        lines = fetch_event_odds(
            league, event_id, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl
        )
        if not lines:
            consecutive_empty += 1
            if consecutive_empty >= CONSECUTIVE_EMPTY_BEFORE_GIVING_UP:
                stats["gaveUp"] = True
            continue

        consecutive_empty = 0
        existing = game.get("lines") or []
        game["lines"] = existing + lines
        game["oddsSource"] = game.get("oddsSource") or "espn-core"
        view_types = game.get("viewTypes") or []
        if "MoneyLine" not in view_types:
            view_types.append("MoneyLine")
        game["viewTypes"] = view_types

        stats["priced"] += 1
        for line in lines:
            book = line.get("sportsbook")
            if book and book not in stats["books"]:
                stats["books"].append(book)

    return stats


# Side-market prices are fetched from the same endpoint as moneylines, so a
# repeated fetch inside this window is pure waste. MLB runs about fifteen games
# a day against a build every thirty minutes; without this that is roughly 720
# requests a day for lines that barely move.
SIDE_MARKET_CACHE_TTL_SECONDS = 60 * 60
_SIDE_MARKET_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

# Where that cache lives between builds. CI starts a fresh interpreter every
# thirty minutes, so an in-process dict never survives to be hit -- the TTL
# above was measuring a window the process never lived to see, and the pass was
# left switched off partly for that reason. Pointed at a file the workflow
# restores and saves, one fetch an hour per game becomes the real rate rather
# than the intended one. Unset outside CI, where an in-process dict is fine.
SIDE_MARKET_CACHE_ENV = "ESPN_SIDE_MARKET_CACHE"


def _cache_path() -> Path | None:
    raw = os.environ.get(SIDE_MARKET_CACHE_ENV, "").strip()
    return Path(raw) if raw else None


def load_side_market_cache() -> int:
    """Seed the in-process cache from disk. Returns how many entries survived.

    Anything past its TTL is dropped on the way in, so a stale file cannot
    pin an old price onto a live board. A malformed or missing file is not an
    error: the pass just fetches, which is what it would have done anyway.
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
    kept = 0
    for key, entry in (payload.get("entries") or {}).items():
        try:
            stamp = float(entry["fetchedAt"])
            lines = entry["lines"]
        except (KeyError, TypeError, ValueError):
            continue
        if not isinstance(lines, list) or now - stamp >= SIDE_MARKET_CACHE_TTL_SECONDS:
            continue
        _SIDE_MARKET_CACHE[key] = (stamp, lines)
        kept += 1
    return kept


def save_side_market_cache() -> int:
    """Write the cache back out. Returns how many entries were written."""
    path = _cache_path()
    if path is None:
        return 0
    now = time.time()
    entries = {
        key: {"fetchedAt": stamp, "lines": lines}
        for key, (stamp, lines) in _SIDE_MARKET_CACHE.items()
        if now - stamp < SIDE_MARKET_CACHE_TTL_SECONDS
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"entries": entries}, indent=None)
    except OSError:
        return 0
    return len(entries)


def clear_side_market_cache() -> None:
    _SIDE_MARKET_CACHE.clear()


def has_priced_market(lines: list[dict[str, Any]], view: str) -> bool:
    """Whether one named market already carries a price on this game.

    Per market, deliberately. The first version of this asked "does ANY side
    market have a price", which is true for almost every MLB game because the
    ESPN summary already prices totals -- so the fetch was skipped and the
    spread, the market that actually needed it, was never asked for. Nought of
    eighty-four MLB runlines priced, with the guard reporting everything fine.
    """
    for line in lines or []:
        if view not in (line.get("viewType") or ""):
            continue
        current = line.get("currentLine") or line.get("openingLine")
        if not isinstance(current, dict):
            continue
        if any(_PRICE_IN_PARENS.search(str(v)) for v in current.values() if v is not None):
            return True
    return False


def has_priced_side_market(lines: list[dict[str, Any]]) -> bool:
    """Both side markets priced. Kept for callers that want the coarse answer."""
    return has_priced_market(lines, "Total") and has_priced_market(lines, "Spread")


def fill_missing_side_market_prices(
    games: list[dict[str, Any]],
    *,
    league: str,
    max_events: int = 40,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
    now: float | None = None,
) -> dict[str, Any]:
    """Price the totals and spreads on games that already have a moneyline.

    fill_missing_moneylines skips any game that is already priced, which is the
    right rule for its own job and the reason MLB side markets were never
    priced at all: SBR gives MLB a moneyline, so ESPN core -- the only source
    that carries total and spread prices -- was never asked about those games.
    The result was 11 priced spreads out of 58, with the rest being baseball
    runlines that could not be valued at all.

    Only the Total and Spread rows are merged. The moneyline is already there
    and a second copy would land in the consensus twice.
    """
    from mlb_predictions import has_moneyline_lines

    stats: dict[str, Any] = {
        "league": league, "considered": 0, "fetched": 0, "priced": 0,
        "books": [], "gaveUp": False, "cached": 0,
    }
    clock = time.time() if now is None else now
    consecutive_empty = 0

    for game in games:
        lines = game.get("lines") or []
        # Needs a moneyline (or the other pass owns it) and needs to be
        # missing a side-market price (or there is nothing to fetch for).
        if not has_moneyline_lines(lines):
            continue
        wanted = [v for v in ("Total", "Spread") if not has_priced_market(lines, v)]
        if not wanted:
            continue
        event_id = game.get("eventId")
        if not event_id:
            continue

        stats["considered"] += 1
        key = f"{league}:{event_id}"
        cached = _SIDE_MARKET_CACHE.get(key)
        if cached and clock - cached[0] < SIDE_MARKET_CACHE_TTL_SECONDS:
            market_lines = cached[1]
            stats["cached"] += 1
        else:
            if stats["fetched"] >= max_events or stats["gaveUp"]:
                continue
            stats["fetched"] += 1
            fetched = fetch_event_odds(
                league, event_id, retries=retries, retry_delay=retry_delay, verify_ssl=verify_ssl
            )
            # Only the markets this game was actually missing. Re-adding a
            # market that is already priced just puts a second copy of the same
            # line into the consensus.
            market_lines = [
                line for line in fetched
                if any(v in (line.get("viewType") or "") for v in wanted)
            ]
            _SIDE_MARKET_CACHE[key] = (clock, market_lines)
            if not market_lines:
                consecutive_empty += 1
                if consecutive_empty >= CONSECUTIVE_EMPTY_BEFORE_GIVING_UP:
                    stats["gaveUp"] = True
                continue
            consecutive_empty = 0

        if not market_lines:
            continue
        game["lines"] = lines + market_lines
        stats["priced"] += 1
        for line in market_lines:
            book = line.get("sportsbook")
            if book and book not in stats["books"]:
                stats["books"].append(book)
    return stats
