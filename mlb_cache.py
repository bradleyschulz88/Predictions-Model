"""Simple thread-safe TTL caches for dashboard responses."""

from __future__ import annotations

import threading
import time
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: float = 180.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, T]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> T | None:
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            created_at, value = entry
            if time.time() - created_at > self.ttl_seconds:
                del self._entries[key]
                return None
            return value

    def get_age_seconds(self, key: str) -> float | None:
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            return time.time() - entry[0]

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._entries[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


DASHBOARD_CACHE: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=180.0)
ENRICHMENT_CACHE: TTLCache[dict[str, Any]] = TTLCache(ttl_seconds=300.0)


def dashboard_cache_key(
    *,
    league: str,
    date_value: str | None,
    view_filter: str,
    source: str,
    fixture: str | None,
    include_odds: bool,
    include_enrichment: bool,
) -> str:
    return "|".join(
        [
            league,
            date_value or "",
            view_filter,
            source,
            fixture or "",
            str(include_odds),
            str(include_enrichment),
        ]
    )
