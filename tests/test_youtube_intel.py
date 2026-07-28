"""Tests for YouTube-sourced pre-game team news.

The leakage guard carries the whole risk here. A recap video published after
the final whistle knows who won; scored as a pre-game feature it would look
like a huge model improvement and be entirely fictional.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from youtube_intel import (
    INTEL_FILE,
    _parse_scores,
    classify_league,
    extract_team_news,
    intel_edge,
    load_intel,
    reset_llm_budget,
)

START = datetime(2026, 7, 28, 23, 5, tzinfo=timezone.utc)


def _record(hours_before_start: float, teams: dict[str, float], video_id: str = "v") -> dict:
    return {
        "videoId": video_id,
        "title": "Preview",
        "channel": "Beat Reporter",
        "publishedAt": (START - timedelta(hours=hours_before_start)).isoformat(),
        "teams": teams,
    }


def _intel(records: list[dict], league: str = "mlb") -> dict:
    return {"builtAt": START.isoformat(), "leagues": {league: records}}


class LeakageGuardTests(unittest.TestCase):
    def test_video_published_before_start_is_used(self) -> None:
        intel = _intel([_record(3, {"New York Yankees": -2.0, "Detroit Tigers": 1.0})])
        edge = intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START)
        self.assertEqual(edge, -3.0)

    def test_video_published_after_start_is_ignored(self) -> None:
        """A recap. Using it would leak the result."""
        intel = _intel([_record(-2, {"New York Yankees": 3.0, "Detroit Tigers": -3.0})])
        self.assertIsNone(
            intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START)
        )

    def test_video_published_exactly_at_start_is_ignored(self) -> None:
        """First pitch is already too late -- the boundary is exclusive."""
        intel = _intel([_record(0, {"New York Yankees": 3.0, "Detroit Tigers": -3.0})])
        self.assertIsNone(
            intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START)
        )

    def test_recap_cannot_rescue_a_pregame_gap(self) -> None:
        """Mixed feed: only the pre-game half may count, and one side alone is not an edge."""
        intel = _intel(
            [
                _record(5, {"New York Yankees": -2.0}, "pre"),
                _record(-1, {"Detroit Tigers": 3.0}, "recap"),
            ]
        )
        self.assertIsNone(
            intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START)
        )

    def test_missing_start_time_drops_the_guard_safely(self) -> None:
        """With no kickoff to compare against, every video is admitted -- so this
        must only ever be called with a real start time. Documented, not silent."""
        intel = _intel([_record(-5, {"A": 1.0, "B": -1.0})])
        self.assertEqual(intel_edge(intel, "mlb", "A", "B", None), 2.0)


class EdgeShapeTests(unittest.TestCase):
    def test_one_sided_coverage_is_not_an_edge(self) -> None:
        intel = _intel([_record(3, {"New York Yankees": -2.0})])
        self.assertIsNone(
            intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START)
        )

    def test_multiple_videos_average(self) -> None:
        intel = _intel(
            [
                _record(6, {"A": -2.0, "B": 0.0}, "one"),
                _record(3, {"A": 0.0, "B": 0.0}, "two"),
            ]
        )
        self.assertEqual(intel_edge(intel, "mlb", "A", "B", START), -1.0)

    def test_team_match_is_case_and_space_insensitive(self) -> None:
        intel = _intel([_record(3, {"  new york YANKEES ": 1.0, "Detroit Tigers": 0.0})])
        self.assertEqual(
            intel_edge(intel, "mlb", "New York Yankees", "Detroit Tigers", START), 1.0
        )

    def test_unknown_league_and_empty_file_are_none(self) -> None:
        intel = _intel([_record(3, {"A": 1.0, "B": 0.0})])
        self.assertIsNone(intel_edge(intel, "nba", "A", "B", START))
        self.assertIsNone(intel_edge({}, "mlb", "A", "B", START))


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_llm_budget()

    def test_parse_scores_handles_fenced_json(self) -> None:
        reply = 'Sure!\n```json\n{"Boston Red Sox": -2, "New York Yankees": 1.5}\n```'
        self.assertEqual(
            _parse_scores(reply), {"Boston Red Sox": -2.0, "New York Yankees": 1.5}
        )

    def test_parse_scores_clamps_to_the_rubric(self) -> None:
        self.assertEqual(_parse_scores('{"A": 99, "B": -99}'), {"A": 3.0, "B": -3.0})

    def test_parse_scores_drops_unparseable_values(self) -> None:
        self.assertEqual(_parse_scores('{"A": "very bad", "B": 2}'), {"B": 2.0})

    def test_parse_scores_survives_junk(self) -> None:
        for reply in (None, "", "no json here", "{not json}"):
            self.assertEqual(_parse_scores(reply), {})

    def test_extraction_is_off_without_a_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(extract_team_news("a transcript", "mlb"), {})

    def test_extraction_falls_back_when_the_model_fails(self) -> None:
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test"}, clear=True), patch(
            "youtube_intel._call_nvidia", return_value=None
        ):
            self.assertEqual(extract_team_news("a transcript", "mlb"), {})


class ClassificationTests(unittest.TestCase):
    def test_wnba_is_not_swallowed_by_nba(self) -> None:
        """"wnba" contains "nba"; order matters."""
        self.assertEqual(classify_league("WNBA tonight: Aces vs Liberty"), "wnba")

    def test_recognises_each_league(self) -> None:
        cases = {
            "MLB preview: Yankees at Tigers": "mlb",
            "NBA Finals breakdown": "nba",
            "NFL Week 1 picks": "nfl",
            "Premier League matchday": "epl",
            "AFL round 20 preview": "afl",
        }
        for text, expected in cases.items():
            self.assertEqual(classify_league(text), expected, text)

    def test_unrecognised_video_is_skipped_not_guessed(self) -> None:
        """A mislabelled video pollutes that league's feature for the whole day."""
        self.assertIsNone(classify_league("My morning routine 2026"))


class LoadTests(unittest.TestCase):
    def test_missing_file_is_empty_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_intel(Path(tmp)), {})

    def test_corrupt_file_is_empty_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / INTEL_FILE).write_text("{ broken", encoding="utf-8")
            self.assertEqual(load_intel(Path(tmp)), {})

    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _intel([_record(3, {"A": 1.0, "B": 0.0})])
            (Path(tmp) / INTEL_FILE).write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                intel_edge(load_intel(Path(tmp)), "mlb", "A", "B", START), 1.0
            )


class FeatureWiringTests(unittest.TestCase):
    def test_is_a_candidate_not_a_live_feature(self) -> None:
        """It must not move a published probability until the ablation says so."""
        from model_fit import ANCHORED_FEATURES, CANDIDATE_FEATURES, STANDALONE_FEATURES

        self.assertIn("videoIntelDiff", CANDIDATE_FEATURES)
        self.assertNotIn("videoIntelDiff", ANCHORED_FEATURES)
        self.assertNotIn("videoIntelDiff", STANDALONE_FEATURES)

    def test_absent_intel_leaves_the_feature_none(self) -> None:
        from model_fit import build_feature_dict

        self.assertIsNone(build_feature_dict({}).get("videoIntelDiff"))

    def test_edge_reaches_the_feature_dict(self) -> None:
        from model_fit import build_feature_dict

        self.assertEqual(
            build_feature_dict({"videoIntelEdge": -1.5}).get("videoIntelDiff"), -1.5
        )


if __name__ == "__main__":
    unittest.main()
