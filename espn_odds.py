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
from typing import Any

from sbr_client import SBRClientError, get_text
from sports_config import LeagueConfig, get_league

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
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) or None
    text = str(value).strip().replace("+", "")
    if not text:
        return None
    if text.upper() in {"EVEN", "EV", "PK"}:
        return 100
    try:
        return int(float(text)) or None
    except ValueError:
        return None


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
