"""Tests for injury severity scoring, which is now deterministic only."""

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
            [_injury("Day-To-Day", "Soreness")], league="mlb", today=TODAY
        )
        heavy = sev.team_injury_severity(
            [_injury("60-Day-IL", "Surgery (Right)"), _injury("60-Day-IL", "Torn ACL")],
            league="mlb",

            today=TODAY,
        )
        self.assertGreater(heavy["score"], light["score"])

    def test_reports_the_per_player_breakdown(self) -> None:
        result = sev.team_injury_severity(
            [_injury(player="Jane Doe")], league="wnba", today=TODAY
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["players"][0]["player"], "Jane Doe")
        self.assertEqual(result["source"], "deterministic")


class NoNvidiaDependencyTests(unittest.TestCase):
    """Nothing in the build may reach for NVIDIA_API_KEY.

    The importance scorer that needed it is gone. Two reasons, and the weaker
    one is that the key was never successfully configured -- so every score this
    project ever published came from the deterministic path regardless.

    The real reason is the ablation: injuryDiff and injurySeverityDiff both made
    walk-forward log loss worse at every sample size measured, 0.6438 to 0.6459
    as they went in. Carrying a metered external dependency, a rate limiter, a
    per-team cache and several hundred lines of key handling to feed a feature
    the data kept declining was not a trade worth making.

    These guard the removal rather than the deletion: it would be easy to
    reintroduce the key by restoring one import.
    """

    BUILD_PATH = (
        "data_providers/injury_severity.py",
        "data_providers/enrich.py",
        "scripts/build_pages_data.py",
    )

    def test_no_build_module_reads_the_key(self) -> None:
        for name in self.BUILD_PATH:
            with self.subTest(name):
                source = (ROOT / name).read_text(encoding="utf-8")
                code = "\n".join(
                    line for line in source.split("\n") if not line.lstrip().startswith("#")
                )
                self.assertNotIn("os.environ.get(\"NVIDIA_API_KEY\")", code)
                self.assertNotIn("environ[\"NVIDIA_API_KEY\"]", code)

    def test_the_workflow_no_longer_passes_the_secret(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertNotIn("secrets.NVIDIA_API_KEY", workflow)

    def test_the_scorer_still_works_without_any_key(self) -> None:
        """The point of the removal: nothing degrades."""
        with mock.patch.dict("os.environ", {}, clear=True):
            result = sev.team_injury_severity(
                [_injury(player="Jane Doe", status="60-Day-IL", detail="Torn ACL")],
                league="mlb",
                team="X",
                today=TODAY,
            )
        self.assertEqual(result["source"], "deterministic")
        self.assertGreater(result["score"], 0.0)

    def test_the_llm_entry_points_are_actually_gone(self) -> None:
        for name in ("api_key", "llm_enabled", "player_importance", "_call_nvidia"):
            self.assertFalse(hasattr(sev, name), f"{name} should have been removed")


if __name__ == "__main__":
    unittest.main()