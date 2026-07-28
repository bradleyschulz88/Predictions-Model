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


if __name__ == "__main__":
    unittest.main()
