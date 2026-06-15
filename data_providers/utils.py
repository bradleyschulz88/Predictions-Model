"""Shared helpers for external sports data providers."""

from __future__ import annotations

import json
import re
from typing import Any

from mlb_cache import PROVIDER_CACHE
from sbr_client import get_text

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


def normalize_team_name(name: str | None) -> str:
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    aliases = {
        "man utd": "manchester united",
        "man city": "manchester city",
        "spurs": "tottenham hotspur",
        "nyc fc": "new york city fc",
        "athletics": "oakland athletics",
        "dodgers": "los angeles dodgers",
        "angels": "los angeles angels",
        "yankees": "new york yankees",
        "mets": "new york mets",
        "white sox": "chicago white sox",
        "cubs": "chicago cubs",
    }
    return aliases.get(cleaned, cleaned)


def team_match_score(candidate: str | None, target: str | None) -> float:
    left = normalize_team_name(candidate)
    right = normalize_team_name(target)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.92
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))
    return overlap


def best_team_match(name: str | None, candidates: dict[str, Any]) -> str | None:
    best_key: str | None = None
    best_score = 0.0
    for key in candidates:
        score = team_match_score(name, key)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key if best_score >= 0.55 else None


def fetch_json(
    url: str,
    *,
    cache_key: str | None = None,
    retries: int = 2,
    retry_delay: float = 0.5,
    verify_ssl: bool = True,
) -> dict[str, Any]:
    if cache_key:
        cached = PROVIDER_CACHE.get(cache_key)
        if cached is not None:
            return cached

    text = get_text(
        url,
        retries=retries,
        retry_delay=retry_delay,
        verify_ssl=verify_ssl,
        user_agent=BROWSER_USER_AGENT,
    )
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}")
    if cache_key:
        PROVIDER_CACHE.set(cache_key, data)
    return data


def stat_map(entries: list[dict[str, Any]] | None) -> dict[str, float | str | None]:
    mapped: dict[str, float | str | None] = {}
    for item in entries or []:
        name = item.get("name")
        if not name:
            continue
        value = item.get("value")
        if value is None:
            value = item.get("displayValue")
        mapped[name] = value
    return mapped


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
