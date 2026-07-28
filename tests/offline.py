"""Block outbound HTTP so tests exercise fixtures rather than live endpoints.

Tests that reached the network were slow, order-dependent and failed in any
sandboxed or offline environment. They also masked what they were really
testing: a payload-shape test should not depend on the MLB Stats API being up.

``get_text`` is the single choke point for every HTTP call in this codebase, but
several modules bind it by name at import time, so each of those namespaces has
to be patched. Patching it (rather than urlopen) also skips the retry/backoff
loop, so blocked calls fail instantly instead of sleeping through three attempts.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbr_client import SBRFetchError  # noqa: E402

# Modules that do `from sbr_client import get_text`; patching sbr_client alone
# would not reach these bindings.
_GET_TEXT_TARGETS = (
    "sbr_client.get_text",
    "espn_client.get_text",
    "espn_enrichment.get_text",
    "data_providers.utils.get_text",
)


def _blocked(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
    url = args[0] if args else kwargs.get("url", "<unknown>")
    raise SBRFetchError(f"Network access is blocked in tests: {url}")


class OfflineTestCase(unittest.TestCase):
    """Base class that fails any test which tries to reach the network.

    Providers already degrade gracefully on fetch errors, so enrichment simply
    yields empty data and the code under test runs on fixtures alone.
    """

    def setUp(self) -> None:
        super().setUp()
        for target in _GET_TEXT_TARGETS:
            patcher = mock.patch(target, side_effect=_blocked)
            patcher.start()
            self.addCleanup(patcher.stop)

        # Provider caches persist across tests and would otherwise leak results
        # from whichever test happened to run first.
        from mlb_cache import PROVIDER_CACHE

        PROVIDER_CACHE.clear()
        self.addCleanup(PROVIDER_CACHE.clear)

        from data_providers.schedule_advanced import clear_rolling_schedule_cache

        clear_rolling_schedule_cache()
        self.addCleanup(clear_rolling_schedule_cache)
