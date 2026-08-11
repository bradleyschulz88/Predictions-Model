"""Tests for injury severity scoring and its optional LLM importance step."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers import injury_severity as sev  # noqa: E402

TODAY = date(2026, 7, 28)


def _injury(status="15-Day-IL", detail="Strain (Left)", player="A Player", return_date=None):
    return {"player": player, "status": status, "detail": detail, "returnDate": return_date}


class DeterministicScoreTests(unittest.TestCase):
    """The old scorer gave nearly every absence the same weight."""

    def test_season_ending_costs_more_than_day_to_day(self) -> None:
        out = sev.deterministic_injury_score(_injury("60-Day-IL", "Surgery (Right)"), today=TODAY)
        minor = sev.deterministic_injury_score(_injury("Day-To-Day", "Soreness"), today=TODAY)
        self.assertGreater(out, minor * 3)

    def test_surgery_outweighs_soreness_at_the_same_status(self) -> None:
        surgery = sev.deterministic_injury_score(_injury("15-Day-IL", "Surgery (Right)"), today=TODAY)
        soreness = sev.deterministic_injury_score(_injury("15-Day-IL", "Soreness (Left)"), today=TODAY)
        self.assertGreater(surgery, soreness)

    def test_illness_is_discounted(self) -> None:
        illness = sev.deterministic_injury_score(_injury("15-Day-IL", "Illness"), today=TODAY)
        strain = sev.deterministic_injury_score(_injury("15-Day-IL", "Strain (Left)"), today=TODAY)
        self.assertLess(illness, strain)

    def test_imminent_return_costs_less_than_a_long_absence(self) -> None:
        soon = sev.deterministic_injury_score(
            _injury(return_date="2026-07-29"), today=TODAY
        )
        distant = sev.deterministic_injury_score(
            _injury(return_date="2026-10-01"), today=TODAY
        )
        self.assertLess(soon, distant)

    def test_unknown_status_still_scores(self) -> None:
        score = sev.deterministic_injury_score(_injury("Mystery", "Unclear"), today=TODAY)
        self.assertGreater(score, 0.0)

    def test_missing_return_date_is_neutral(self) -> None:
        self.assertAlmostEqual(sev._return_multiplier(None, TODAY), 1.0)

    def test_unparseable_return_date_is_neutral(self) -> None:
        self.assertAlmostEqual(sev._return_multiplier("not a date", TODAY), 1.0)


class TeamSeverityTests(unittest.TestCase):
    def test_empty_list_scores_zero(self) -> None:
        result = sev.team_injury_severity([], league="mlb")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["source"], "none")

    def test_none_is_handled(self) -> None:
        self.assertEqual(sev.team_injury_severity(None, league="mlb")["score"], 0.0)

    def test_more_serious_squad_scores_higher(self) -> None:
        light = sev.team_injury_severity(
            [_injury("Day-To-Day", "Soreness")], league="mlb", use_llm=False, today=TODAY
        )
        heavy = sev.team_injury_severity(
            [_injury("60-Day-IL", "Surgery (Right)"), _injury("60-Day-IL", "Torn ACL")],
            league="mlb",
            use_llm=False,
            today=TODAY,
        )
        self.assertGreater(heavy["score"], light["score"])

    def test_reports_the_per_player_breakdown(self) -> None:
        result = sev.team_injury_severity(
            [_injury(player="Jane Doe")], league="wnba", use_llm=False, today=TODAY
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["players"][0]["player"], "Jane Doe")
        self.assertEqual(result["source"], "deterministic")


class LlmImportanceTests(unittest.TestCase):
    """The LLM step is optional and must never be able to break a build."""

    def setUp(self) -> None:
        from mlb_cache import PROVIDER_CACHE

        PROVIDER_CACHE.clear()
        self.addCleanup(PROVIDER_CACHE.clear)

    def test_disabled_without_an_api_key(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(sev.llm_enabled())
            self.assertEqual(sev.player_importance([_injury()], league="mlb", team="X"), {})

    def test_network_failure_falls_back_to_deterministic(self) -> None:
        with mock.patch.dict("os.environ", {"NVIDIA_API_KEY": "k"}), mock.patch.object(
            sev, "_call_nvidia", return_value=None
        ):
            result = sev.team_injury_severity(
                [_injury(player="Jane Doe")], league="mlb", team="X", today=TODAY
            )
        self.assertEqual(result["source"], "deterministic")
        self.assertGreater(result["score"], 0.0)

    def test_importance_scales_the_cost(self) -> None:
        with mock.patch.dict("os.environ", {"NVIDIA_API_KEY": "k"}), mock.patch.object(
            sev, "_call_nvidia", return_value='{"Star Player": 3, "Bench Guy": 0}'
        ):
            result = sev.team_injury_severity(
                [_injury(player="Star Player"), _injury(player="Bench Guy")],
                league="nba",
                team="X",
                today=TODAY,
            )
        by_name = {item["player"]: item for item in result["players"]}
        self.assertGreater(by_name["Star Player"]["cost"], by_name["Bench Guy"]["cost"])
        self.assertEqual(result["source"], "llm")

    def test_parses_a_reply_wrapped_in_prose_and_fences(self) -> None:
        reply = 'Sure! Here you go:\n```json\n{"A Player": 2}\n```\nHope that helps.'
        self.assertEqual(sev._parse_importance(reply), {"a player": 1.5})

    def test_unparseable_reply_yields_no_importance(self) -> None:
        self.assertEqual(sev._parse_importance("I cannot help with that."), {})
        self.assertEqual(sev._parse_importance(None), {})

    def test_out_of_range_ratings_are_clamped(self) -> None:
        scores = sev._parse_importance('{"A": 99, "B": -5}')
        self.assertEqual(scores["a"], 2.0)
        self.assertEqual(scores["b"], 0.5)

    def test_non_numeric_ratings_are_skipped(self) -> None:
        self.assertEqual(sev._parse_importance('{"A": "very important"}'), {})

    def test_results_are_cached_per_team(self) -> None:
        call = mock.Mock(return_value='{"A Player": 2}')
        with mock.patch.dict("os.environ", {"NVIDIA_API_KEY": "k"}), mock.patch.object(
            sev, "_call_nvidia", call
        ):
            for _ in range(3):
                sev.player_importance([_injury()], league="mlb", team="X")
        self.assertEqual(call.call_count, 1)

    def test_requests_are_deterministic(self) -> None:
        """Two builds half an hour apart must not score the same slate differently."""
        captured = {}

        def fake_urlopen(request, timeout=None):  # noqa: ANN001
            captured["body"] = request.data.decode("utf-8")
            raise OSError("stop here")

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            sev._call_nvidia("prompt", "key")
        self.assertIn('"temperature": 0.0', captured["body"])



class RateLimitTests(unittest.TestCase):
    """A full build scores ~80 teams against a 40 requests/minute free tier."""

    def setUp(self) -> None:
        sev.reset_llm_budget()
        self.addCleanup(sev.reset_llm_budget)

    def test_calls_are_spaced_to_stay_under_the_limit(self) -> None:
        slept = []
        with mock.patch.object(sev.time, "sleep", slept.append), mock.patch(
            "urllib.request.urlopen", side_effect=OSError("stop")
        ):
            sev._call_nvidia("a", "k")
            sev._call_nvidia("b", "k")
        # The second call waits; without the throttle both fire instantly.
        self.assertTrue(any(value > 0 for value in slept))

    def test_rate_limit_is_retried_once_then_gives_up(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        with mock.patch.object(sev.time, "sleep"), mock.patch(
            "urllib.request.urlopen", side_effect=error
        ) as call:
            self.assertIsNone(sev._call_nvidia("a", "k"))
        self.assertEqual(call.call_count, 2)

    def test_other_http_errors_are_not_retried(self) -> None:
        import urllib.error

        error = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
        with mock.patch.object(sev.time, "sleep"), mock.patch(
            "urllib.request.urlopen", side_effect=error
        ) as call:
            self.assertIsNone(sev._call_nvidia("a", "k"))
        self.assertEqual(call.call_count, 1)

    def test_per_run_budget_stops_a_runaway_slate(self) -> None:
        sev._calls_made = sev.MAX_CALLS_PER_RUN
        with mock.patch("urllib.request.urlopen") as call:
            self.assertIsNone(sev._call_nvidia("a", "k"))
        call.assert_not_called()

    def test_a_blocked_network_degrades_to_deterministic(self) -> None:
        """Exactly what happened against the real endpoint from a sandbox."""
        import urllib.error

        with mock.patch.dict("os.environ", {"NVIDIA_API_KEY": "k"}), mock.patch.object(
            sev.time, "sleep"
        ), mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Tunnel 403")):
            result = sev.team_injury_severity(
                [_injury(player="Aaron Judge")], league="mlb", team="NYY", today=TODAY
            )
        self.assertEqual(result["source"], "deterministic")
        self.assertGreater(result["score"], 0.0)

class InjuryScorerReportTests(unittest.TestCase):
    """A green build is not evidence the NVIDIA key works.

    The key is optional: without it team_injury_severity falls back to a
    deterministic score and the build succeeds identically. So an expired key or
    an exhausted quota degrades in complete silence unless something says which
    path ran.
    """

    def _payload(self, *pairs: tuple[str, str]) -> dict:
        return {
            "games": [
                {
                    "enrichment": {
                        "homeInjurySeverity": {"source": home},
                        "awayInjurySeverity": {"source": away},
                    }
                }
                for home, away in pairs
            ]
        }

    def _report(self, payloads: dict) -> str:
        import io
        from contextlib import redirect_stdout

        import sys

        sys.path.insert(0, "scripts")
        from build_pages_data import report_injury_scorer

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report_injury_scorer(payloads)
        return buffer.getvalue()

    def test_reports_llm_when_the_key_works(self) -> None:
        out = self._report({"mlb": self._payload(("llm", "llm"), ("llm", "deterministic"))})
        self.assertIn("LLM rated 3/4", out)
        self.assertIn("is working", out)

    def test_reports_fallback_when_the_key_is_missing(self) -> None:
        """No key is a supported mode, so it must not read as a failure."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_API_KEY", None)
            out = self._report({"mlb": self._payload(("deterministic", "deterministic"))})
        self.assertIn("deterministic on all 2", out)
        self.assertIn("not set", out)
        self.assertNotIn("::warning", out)

    def test_a_key_that_is_set_but_failing_warns_with_the_reason(self) -> None:
        """The case that sent the user to rotate a key twice: name the cause."""
        import os
        from unittest import mock

        from data_providers import injury_severity

        injury_severity._note_failure("HTTP 401, the key was rejected")
        try:
            with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}):
                out = self._report(
                    {"mlb": self._payload(("deterministic", "deterministic"))}
                )
        finally:
            injury_severity.reset_failure()
        self.assertIn("::warning title=Injury scorer::", out)
        self.assertIn("HTTP 401", out)
        self.assertIn("the key is set", out)

    def test_no_injuries_is_not_reported_as_failure(self) -> None:
        """"none" means nobody was hurt, which says nothing about the key."""
        out = self._report({"mlb": self._payload(("none", "none"))})
        self.assertIn("no teams with injuries", out)
        self.assertNotIn("absent", out)

    def test_empty_build_is_safe(self) -> None:
        self.assertIn("no teams", self._report({}))


class FailureReportingTests(unittest.TestCase):
    """A rotated key that still fails must say WHY, not repeat one vague string."""

    def setUp(self) -> None:
        from data_providers import injury_severity

        self.module = injury_severity
        injury_severity.reset_failure()

    def _http_error(self, code: int, reason: str, body: bytes = b"") -> Exception:
        import io
        import urllib.error

        return urllib.error.HTTPError("http://x", code, reason, {}, io.BytesIO(body))

    def test_a_rejected_key_says_to_regenerate_it(self) -> None:
        message = self.module._describe_http_error(self._http_error(401, "Unauthorized"))
        self.assertIn("401", message)
        self.assertIn("regenerate", message)

    def test_a_model_entitlement_failure_says_not_to_rotate(self) -> None:
        """403 is the case where rotating the key wastes the user's time."""
        message = self.module._describe_http_error(self._http_error(403, "Forbidden"))
        self.assertIn("NOT a stale key", message)

    def test_quota_is_distinguished_from_a_bad_key(self) -> None:
        quota = self.module._describe_http_error(self._http_error(429, "Too Many"))
        rejected = self.module._describe_http_error(self._http_error(401, "Unauthorized"))
        self.assertNotEqual(quota, rejected)
        self.assertIn("quota", quota)

    def test_a_retired_model_names_the_override(self) -> None:
        message = self.module._describe_http_error(self._http_error(404, "Not Found"))
        self.assertIn("NVIDIA_INJURY_MODEL", message)

    def test_the_reason_survives_for_the_build_report(self) -> None:
        self.assertIsNone(self.module.last_failure())
        self.module._note_failure("HTTP 401, the key was rejected")
        self.assertIn("401", self.module.last_failure())

    def test_an_unreachable_api_is_not_reported_as_a_bad_key(self) -> None:
        import urllib.error

        original = self.module.urllib.request.urlopen

        def boom(*args, **kwargs):
            raise urllib.error.URLError("proxy refused")

        self.module.urllib.request.urlopen = boom
        try:
            self.module._calls_made = 0
            self.assertIsNone(self.module._call_nvidia("prompt", "nvapi-test"))
        finally:
            self.module.urllib.request.urlopen = original
        reason = self.module.last_failure()
        self.assertIn("could not reach", reason)
        self.assertNotIn("regenerate", reason)


class ApiKeyHygieneTests(unittest.TestCase):
    """A pasted key carries a newline; urllib refuses to send it."""

    def setUp(self) -> None:
        from data_providers import injury_severity

        self.module = injury_severity
        injury_severity.reset_failure()

    def _with(self, value):
        import os
        from unittest import mock

        if value is None:
            ctx = mock.patch.dict(os.environ, {}, clear=False)
            ctx.__enter__()
            os.environ.pop("NVIDIA_API_KEY", None)
            return ctx
        return mock.patch.dict(os.environ, {"NVIDIA_API_KEY": value})

    def test_a_trailing_newline_is_stripped(self) -> None:
        """The exact production failure: 0 of 34 teams, masked value ending in \\."""
        with self._with("nvapi-abc123\n"):
            self.assertEqual(self.module.api_key(), "nvapi-abc123")
            self.assertTrue(self.module.llm_enabled())

    def test_surrounding_spaces_are_stripped(self) -> None:
        with self._with("  nvapi-abc123\t"):
            self.assertEqual(self.module.api_key(), "nvapi-abc123")

    def test_a_stripped_key_makes_a_valid_header(self) -> None:
        """The regression that matters: the header must be constructible."""
        import urllib.request

        with self._with("nvapi-abc123\n"):
            key = self.module.api_key()
        request = urllib.request.Request(
            "https://example.invalid", headers={"Authorization": f"Bearer {key}"}
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer nvapi-abc123")

    def test_a_raw_newline_key_would_have_failed(self) -> None:
        """Documents why stripping is required, not merely tidy.

        Request() accepts the bad value; http.client rejects it when the header
        is actually written, which is why the production symptom was a generic
        "could not reach the API" rather than anything mentioning the key.
        """
        import http.client

        connection = http.client.HTTPConnection("example.invalid")
        connection.putrequest("POST", "/", skip_host=True, skip_accept_encoding=True)
        with self.assertRaises(ValueError) as caught:
            connection.putheader("Authorization", "Bearer nvapi-abc123\n")
        self.assertIn("Invalid header value", str(caught.exception))

    def test_interior_whitespace_is_refused_with_its_own_reason(self) -> None:
        """Two keys end to end: joining them makes a longer wrong key, not a right one."""
        doubled = "nvapi-abc123 nvapi-def456"
        with self._with(doubled):
            self.assertIsNone(self.module.api_key())
        self.assertIn("single line", self.module.last_failure())

    def test_one_key_split_across_lines_is_rejoined(self) -> None:
        """A wrapped paste is recoverable, and failing a build over it is silly.

        The discriminator is the prefix count: one key broken in half still has
        a single `nvapi-`, two keys pasted end to end have two.
        """
        with self._with("nvapi-abc123\ndef456"):
            self.assertEqual(self.module.api_key(), "nvapi-abc123def456")

    def test_a_rejoined_key_still_says_to_fix_the_secret(self) -> None:
        """Recovered is not correct. It must not go quiet."""
        with self._with("nvapi-abc123\ndef456"):
            self.module.api_key()
        reason = self.module.last_failure()
        self.assertIn("re-paste", reason)
        self.assertIn("single line", reason)
    def test_whitespace_only_is_refused(self) -> None:
        with self._with("   "):
            self.assertIsNone(self.module.api_key())
        self.assertIn("only whitespace", self.module.last_failure())

    def test_an_unset_key_reports_nothing(self) -> None:
        ctx = self._with(None)
        try:
            self.assertIsNone(self.module.api_key())
        finally:
            ctx.__exit__(None, None, None)
        self.assertIsNone(self.module.last_failure())

    def test_the_placeholder_from_nvidias_own_snippet_is_caught(self) -> None:
        """NVIDIA's quick-start reads `api_key = "$NVIDIA_API_KEY"`.

        That is shell syntax in a Python string literal: nothing expands it, so
        the fifteen characters go out as the credential and come back 401 --
        indistinguishable from a revoked key, which sends you to rotate a key
        that was never the problem.
        """
        for placeholder in ("$NVIDIA_API_KEY", "${NVIDIA_API_KEY}", "%NVIDIA_API_KEY%"):
            with self.subTest(placeholder):
                self.module.reset_failure()
                with self._with(placeholder):
                    self.assertIsNone(self.module.api_key())
                self.assertIn("literal text", self.module.last_failure())

    def test_the_advice_names_the_actual_fix(self) -> None:
        """Not "rotate the key" -- the key does not exist yet."""
        with self._with("$NVIDIA_API_KEY"):
            self.module.api_key()
        self.assertIn("nvapi-", self.module.last_failure())

    def test_a_real_key_containing_a_dollar_sign_is_left_alone(self) -> None:
        """The check is for a value that is only a variable reference."""
        for key in ("nvapi-ab$cd", "$nvapi-abc-def", "nvapi-$"):
            with self.subTest(key):
                with self._with(key):
                    self.assertEqual(self.module.api_key(), key)


class KeyShapeReportTests(unittest.TestCase):
    """A rejected secret has to describe itself, because nobody else can see it.

    Once saved, a GitHub secret is write-only -- the build is the only thing
    that can look at the value. "Contains a space or line break" was true and
    useless: a key wrapped across two lines, two keys pasted end to end, and a
    whole line of source code in the field all produce it, and all three need a
    different fix. So the reason now reports the value's shape.

    Nothing reported is the key. A length, an offset, a character class, and
    whether the published `nvapi-` prefix is present.
    """

    def setUp(self) -> None:
        from data_providers import injury_severity

        self.module = injury_severity
        injury_severity.reset_failure()

    def _reason(self, value):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": value}):
            self.module.api_key()
        return self.module.last_failure()

    def test_it_names_the_kind_of_whitespace(self) -> None:
        self.assertIn("line break", self._reason("nvapi-a\nb nvapi-c"))
        self.module.reset_failure()
        self.assertIn("a space", self._reason("nvapi-a b nvapi-c"))

    def test_a_non_breaking_space_is_called_out_by_name(self) -> None:
        """The signature of copying out of a rendered page rather than a field."""
        self.assertIn("non-breaking", self._reason("nvapi-a\xa0b nvapi-c"))

    def test_it_gives_a_length_and_a_position(self) -> None:
        reason = self._reason("nvapi-abc def nvapi-x")
        self.assertIn("21 characters", reason)
        self.assertIn("position 9", reason)

    def test_two_keys_are_named_as_two_keys(self) -> None:
        reason = self._reason("nvapi-abc123 nvapi-def456")
        self.assertIn("two keys, not one", reason)

    def test_a_value_that_is_not_a_key_at_all_says_so(self) -> None:
        """`api_key = "nvapi-..."` pasted whole is the obvious way to get here."""
        reason = self._reason('api_key = "nvapi-abc123"')
        self.assertIn("does not start with", reason)

    def test_the_report_never_contains_the_key_itself(self) -> None:
        """The whole point: diagnosable without being readable."""
        reason = self._reason("nvapi-SECRETPART1 nvapi-SECRETPART2")
        self.assertNotIn("SECRETPART", reason)



if __name__ == "__main__":
    unittest.main()
