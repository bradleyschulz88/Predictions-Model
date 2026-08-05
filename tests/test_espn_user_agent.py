"""Every ESPN request must identify itself honestly.

On 2026-08-04 the whole board went to FEED ERROR because Akamai started
returning 403 to requests claiming to be Mozilla. The scoreboard fetch was
sending "Mozilla/5.0 (compatible; MLB-SBR-Client/1.0)" purely because
fetch_scoreboard never passed a user_agent and inherited the SBR default.

Nothing failed loudly: the build succeeded, published empty slates, and the
outage was only visible on the site. These tests pin the header down so a
future caller cannot quietly reintroduce a browser-shaped UA.
"""

from __future__ import annotations

import unittest
from unittest import mock

import data_providers.espn_advanced as espn_advanced
import espn_client
import espn_enrichment
import espn_odds
from espn_client import ESPN_USER_AGENT


class EspnUserAgentValueTest(unittest.TestCase):
    def test_does_not_claim_to_be_a_browser(self) -> None:
        # The exact string Akamai rejects. Anything Mozilla-shaped is refused
        # whether or not it is a full browser UA, so assert on the token.
        self.assertNotIn("mozilla", ESPN_USER_AGENT.lower())

    def test_carries_a_contact_url(self) -> None:
        # Probed 5/5: "EdgeBoard/1.0" alone is refused, "EdgeBoard/1.0 (+url)"
        # is accepted. The comment is load-bearing, not decoration.
        self.assertIn("(+http", ESPN_USER_AGENT)


class EspnCallSiteTest(unittest.TestCase):
    """Each ESPN fetcher must pass the honest UA rather than inherit a default."""

    def _captured_user_agent(self, module: object, call: object) -> str | None:
        seen: dict[str, str | None] = {}

        def fake_get_text(url: str, **kwargs: object) -> str:
            seen["user_agent"] = kwargs.get("user_agent")  # type: ignore[assignment]
            return "{}"

        with mock.patch.object(module, "get_text", fake_get_text):
            call()
        return seen.get("user_agent")

    def test_scoreboard_sends_the_espn_user_agent(self) -> None:
        agent = self._captured_user_agent(
            espn_client, lambda: espn_client.fetch_scoreboard("mlb", "2026-08-05")
        )
        self.assertEqual(agent, ESPN_USER_AGENT)

    def test_summary_sends_the_espn_user_agent(self) -> None:
        agent = self._captured_user_agent(
            espn_enrichment,
            lambda: espn_enrichment.fetch_event_summary("401815776", league="mlb"),
        )
        self.assertEqual(agent, ESPN_USER_AGENT)

    def test_core_odds_sends_the_espn_user_agent(self) -> None:
        agent = self._captured_user_agent(
            espn_odds, lambda: espn_odds.fetch_event_odds("mlb", "401815776")
        )
        self.assertEqual(agent, ESPN_USER_AGENT)

    def test_team_directory_sends_the_espn_user_agent(self) -> None:
        seen: dict[str, object] = {}

        def fake_fetch_json(url: str, **kwargs: object) -> dict[str, object]:
            seen["user_agent"] = kwargs.get("user_agent")
            return {}

        with mock.patch.object(espn_advanced, "fetch_json", fake_fetch_json):
            espn_advanced.fetch_espn_team_directory("mlb")
        self.assertEqual(seen.get("user_agent"), ESPN_USER_AGENT)


if __name__ == "__main__":
    unittest.main()
