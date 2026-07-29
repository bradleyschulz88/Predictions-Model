"""Season-series parsing, against the wording ESPN actually uses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_providers.derived import series_win_pct  # noqa: E402
from mlb_predictions import _h2h_diff  # noqa: E402

# The exact payload a production build reported. Of 25 games, 25 carried a
# season series and a score, and 0 resolved -- because the summary names the
# club by ABBREVIATION and the matcher only tried the name and the nickname.
REAL = {"summary": "TB wins series 5-1", "seriesScore": "5-1"}


class AbbreviationMatchingTests(unittest.TestCase):
    def test_the_real_espn_wording_resolves(self) -> None:
        self.assertAlmostEqual(series_win_pct(REAL, "Tampa Bay Rays", "TB"), 5 / 6)

    def test_without_the_abbreviation_it_cannot_match(self) -> None:
        """Documents the old behaviour, so the regression is unmistakable."""
        self.assertIsNone(series_win_pct(REAL, "Tampa Bay Rays"))

    def test_the_unnamed_opponent_stays_unknown_here(self) -> None:
        """Only one club is named; the other is inferred later by complement."""
        self.assertIsNone(series_win_pct(REAL, "Boston Red Sox", "BOS"))

    def test_a_bare_abbreviation_is_word_matched(self) -> None:
        """"TB" must not be found inside another token."""
        series = {"summary": "TBR wins series 5-1", "seriesScore": "5-1"}
        self.assertIsNone(series_win_pct(series, "Tampa Bay Rays", "TB"))

    def test_the_club_named_first_holds_the_first_number(self) -> None:
        series = {"summary": "BOS wins series 4-2", "seriesScore": "4-2"}
        self.assertAlmostEqual(series_win_pct(series, "Boston Red Sox", "BOS"), 4 / 6)

    def test_full_names_still_work_for_other_leagues(self) -> None:
        series = {"summary": "Arsenal lead series 2-1", "seriesScore": "2-1"}
        self.assertAlmostEqual(series_win_pct(series, "Arsenal", None), 2 / 3)

    def test_longer_tokens_win_over_shorter_ones(self) -> None:
        """"Chicago White Sox" must not be resolved by matching "Sox"."""
        series = {"summary": "Chicago White Sox wins series 4-2", "seriesScore": "4-2"}
        self.assertAlmostEqual(series_win_pct(series, "Chicago White Sox", "CHW"), 4 / 6)

    def test_a_tied_series_names_nobody(self) -> None:
        series = {"summary": "Series tied 1-1", "seriesScore": "1-1"}
        self.assertIsNone(series_win_pct(series, "Tampa Bay Rays", "TB"))

    def test_missing_pieces_are_not_guessed(self) -> None:
        self.assertIsNone(series_win_pct(None, "Tampa Bay Rays", "TB"))
        self.assertIsNone(series_win_pct({"summary": "TB wins series"}, "Tampa Bay Rays", "TB"))
        self.assertIsNone(series_win_pct({"seriesScore": "5-1"}, "Tampa Bay Rays", "TB"))
        self.assertIsNone(series_win_pct(REAL, None, None))


class FeatureResolutionTests(unittest.TestCase):
    """The end the model cares about: does h2hDiff come out non-None."""

    def _diff(self, home: str, home_abbr: str, away: str, away_abbr: str) -> float | None:
        return _h2h_diff({
            "headToHead": {
                "homeSeriesWinPct": series_win_pct(REAL, home, home_abbr),
                "awaySeriesWinPct": series_win_pct(REAL, away, away_abbr),
                "summary": REAL["summary"],
                "seriesScore": REAL["seriesScore"],
            }
        })

    def test_the_series_leader_at_home_gives_a_positive_edge(self) -> None:
        diff = self._diff("Tampa Bay Rays", "TB", "Boston Red Sox", "BOS")
        self.assertIsNotNone(diff, "h2hDiff was None on every game before this")
        self.assertGreater(diff, 0)

    def test_the_series_leader_away_gives_a_negative_edge(self) -> None:
        diff = self._diff("Boston Red Sox", "BOS", "Tampa Bay Rays", "TB")
        self.assertIsNotNone(diff)
        self.assertLess(diff, 0)

    def test_the_two_orientations_are_mirror_images(self) -> None:
        home_led = self._diff("Tampa Bay Rays", "TB", "Boston Red Sox", "BOS")
        away_led = self._diff("Boston Red Sox", "BOS", "Tampa Bay Rays", "TB")
        self.assertAlmostEqual(home_led, -away_led)


class WiringTests(unittest.TestCase):
    def test_enrichment_passes_the_abbreviation(self) -> None:
        """The fix is inert unless the abbreviation reaches the matcher."""
        source = (ROOT / "data_providers" / "enrich.py").read_text(encoding="utf-8")
        self.assertIn('series_win_pct(series, home_team, game.get("homeAbbr"))', source)
        self.assertIn('series_win_pct(series, away_team, game.get("awayAbbr"))', source)


if __name__ == "__main__":
    unittest.main()
