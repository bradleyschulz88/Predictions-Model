"""Shared client for fetching structured data from SportsBookReview MLB pages."""

from __future__ import annotations

import html
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MLB-SBR-Client/1.0)"
NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)

SBR_ODDS_BASE = "https://www.sportsbookreview.com/betting-odds/mlb-baseball/"
SBR_MATCHUPS_BASE = "https://www.sportsbookreview.com/scores/mlb-baseball/matchups/"
SBR_MATCHUP_BASE = "https://www.sportsbookreview.com/scores/mlb-baseball/matchup/"


class SBRClientError(Exception):
    """Base error for SBR client operations."""


class SBRFetchError(SBRClientError):
    """HTTP or network failure while fetching a page."""


class SBRParseError(SBRClientError):
    """Page fetched but __NEXT_DATA__ JSON could not be extracted or parsed."""


def build_odds_url(date: str | None = None) -> str:
    if date:
        return f"{SBR_ODDS_BASE}?date={date}"
    return SBR_ODDS_BASE


def build_matchups_url() -> str:
    return SBR_MATCHUPS_BASE


def build_matchup_url(matchup_id: int | str) -> str:
    return f"{SBR_MATCHUP_BASE}{matchup_id}"


def get_text(
    url: str,
    *,
    timeout: int = 30,
    retries: int = 3,
    retry_delay: float = 1.0,
    user_agent: str = DEFAULT_USER_AGENT,
    verify_ssl: bool = True,
) -> str:
    """Fetch page HTML with retries and basic rate-limit spacing."""
    last_error: Exception | None = None
    context = None if verify_ssl else ssl._create_unverified_context()

    for attempt in range(retries):
        if attempt > 0:
            time.sleep(retry_delay * attempt)

        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            last_error = SBRFetchError(f"HTTP {exc.code} for {url}")
            continue
        except urllib.error.URLError as exc:
            last_error = SBRFetchError(f"Network error for {url}: {exc.reason}")
            continue

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            last_error = SBRFetchError(f"Invalid UTF-8 response from {url}")
            continue

    assert last_error is not None
    raise last_error


def extract_next_data(html_text: str) -> dict[str, Any]:
    """Parse __NEXT_DATA__ JSON from a Next.js HTML page."""
    match = NEXT_DATA_PATTERN.search(html_text)
    if not match:
        raise SBRParseError("__NEXT_DATA__ script tag not found in page HTML")

    payload = html.unescape(match.group(1))
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SBRParseError(f"Invalid JSON in __NEXT_DATA__: {exc}") from exc


def fetch_next_data(url: str, **kwargs: Any) -> dict[str, Any]:
    """Fetch a URL and return parsed __NEXT_DATA__ JSON."""
    html_text = get_text(url, **kwargs)
    return extract_next_data(html_text)


def get_page_props(url: str, **kwargs: Any) -> dict[str, Any]:
    """Fetch a URL and return the Next.js pageProps object."""
    data = fetch_next_data(url, **kwargs)
    page_props = data.get("props", {}).get("pageProps")
    if not isinstance(page_props, dict):
        raise SBRParseError("pageProps missing or not an object in __NEXT_DATA__")
    return page_props


def get_game_rows(page_props: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract game rows from an odds listing pageProps object."""
    odds_tables = page_props.get("oddsTables")
    if not odds_tables or not isinstance(odds_tables, list):
        raise SBRParseError("oddsTables missing or empty in pageProps")

    first_table = odds_tables[0]
    if not isinstance(first_table, dict):
        raise SBRParseError("oddsTables[0] is not an object")

    model = first_table.get("oddsTableModel")
    if not isinstance(model, dict):
        raise SBRParseError("oddsTableModel missing in oddsTables[0]")

    rows = model.get("gameRows")
    if not isinstance(rows, list):
        raise SBRParseError("gameRows missing in oddsTableModel")

    return rows


def get_matchup(page_props: dict[str, Any]) -> dict[str, Any]:
    """Extract matchup object from a matchup detail pageProps object."""
    matchup_model = page_props.get("matchupModel")
    if not isinstance(matchup_model, dict):
        raise SBRParseError("matchupModel missing in pageProps")

    matchup = matchup_model.get("matchup")
    if not isinstance(matchup, dict):
        raise SBRParseError("matchup missing in matchupModel")

    return matchup
