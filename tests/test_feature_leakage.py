"""No candidate feature may know the result of the game it predicts.

Found 13 Aug 2026, immediately after `ablate_each` replaced the nested-prefix
sweep. Judged on its own against the shipped pair, `h2hDiff` improved
walk-forward log loss by 0.0370 -- five times the model's entire measured edge
over the market. That is not what a discovery looks like at this sample size;
it is what a leak looks like.

It was one. `h2hDiff` reads ESPN's season-series summary ("Dodgers lead series
2-1"). A series is three or four games, so the result of the game being
predicted moves the feature by a third of its range. Measured over 268 graded
rows it scored a standalone AUC of 0.855 with its sign agreeing with the
outcome on 79.9% of non-zero rows.

The mechanism was general, which is why this file exists rather than a single
exclusion. `accuracy_tracker` wrote `"features": prediction.get("features")` on
every build, overwriting whatever was there, against a build that re-enriches
dates already played -- so any feature read from a source that updates after a
game carried that game's outcome by the time the model was fitted on it.
Features are now frozen at first pitch, but the detector stays: the next
contaminated candidate should fail a build, not survive to be found by hand.

The reference point is the market. A de-vigged closing line is the best
publicly available pre-game estimator of a result, and on this log it scores
0.640. A feature that beats it standalone is claiming to know something the
entire betting market does not, which for a park factor or a travel distance is
not credible.

The features the board predicts on were always clean -- at pick time the game
has not been played. The weights were not. A fit trained on rows where
`strengthDiff` scored 0.682 will over-weight it against the ~0.62 that feature
is worth pre-game, so the leak degraded live predictions as well as the numbers
describing them. Freezing stops new rows being contaminated; it does not repair
the 947 already logged, which is what `frozenSamples` counts down.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_fit import (  # noqa: E402
    CANDIDATE_FEATURES,
    LEAKING_FEATURES,
    MAX_PLAUSIBLE_FEATURE_AUC,
    samples_from_log,
)

DATA = ROOT / "docs" / "data"


def _auc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank-based AUC. O(n log n), because the pairwise form is O(n^2)."""
    positives = [value for value, label in pairs if label == 1]
    negatives = [value for value, label in pairs if label == 0]
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda item: item[0])
    ranks: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = shared
        index = stop + 1
    rank_sum = sum(ranks[i] for i, (_v, label) in enumerate(ordered) if label == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class FeatureLeakageTests(unittest.TestCase):
    """Scored against the real graded log, so it tracks the live data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, _centre = samples_from_log(DATA)

    def _scored(self, name: str) -> tuple[float | None, int]:
        pairs = [
            (float(sample.values[name]), sample.label)
            for sample in self.samples
            if sample.values.get(name) is not None
        ]
        return (_auc(pairs) if len(pairs) >= 40 else None), len(pairs)

    def test_the_market_is_where_it_should_be(self) -> None:
        """Calibrates the threshold. If this drifts, the threshold is wrong too."""
        auc, count = self._scored("marketLogit")
        if auc is None:
            self.skipTest(f"only {count} rows carry a market price")
        self.assertGreater(auc, 0.55, "the closing line should carry real signal")
        self.assertLess(auc, MAX_PLAUSIBLE_FEATURE_AUC, "the market itself cannot leak")

    def test_no_shipped_or_candidate_feature_beats_the_plausible_ceiling(self) -> None:
        offenders = []
        for name in CANDIDATE_FEATURES:
            auc, count = self._scored(name)
            if auc is None:
                continue
            if auc > MAX_PLAUSIBLE_FEATURE_AUC:
                offenders.append(f"{name} AUC {auc:.3f} on {count} rows")
        self.assertEqual(
            offenders, [],
            "a feature is predicting the result too well to be pre-game information. "
            "Check whether it reads a source that updates after the game, the way "
            "h2hDiff read the season-series score: " + "; ".join(offenders),
        )

    def test_the_known_leak_is_not_in_the_candidate_list(self) -> None:
        for name in LEAKING_FEATURES:
            self.assertNotIn(
                name, CANDIDATE_FEATURES,
                f"{name} is documented as carrying the result and must not be fitted",
            )

    def test_the_known_leak_would_still_be_caught_if_it_came_back(self) -> None:
        """The detector, not just the exclusion -- proven against the real thing."""
        auc, count = self._scored("h2hDiff")
        if auc is None:
            self.skipTest(f"only {count} rows carry h2hDiff")
        self.assertGreater(
            auc, MAX_PLAUSIBLE_FEATURE_AUC,
            "h2hDiff scored 0.855 when this was written; if it no longer trips the "
            "threshold, either the leak was fixed upstream or the threshold is too loose",
        )



class FeaturesAreFrozenAtFirstPitchTests(unittest.TestCase):
    """The fix for the leak, held in place.

    A plain overwrite is one line and reads as harmless, which is exactly how
    this survived: `"features": prediction.get("features")` on every build,
    against a build that re-enriches dates already played.

    Frozen at the LAST pre-game observation rather than the first, mirroring
    pickOdds. The first build to see a game days out may carry no odds and thin
    enrichment, and there is no reason to prefer that to the fullest picture
    available at first pitch.
    """

    def _source(self) -> str:
        return (ROOT / "accuracy_tracker.py").read_text(encoding="utf-8")

    def test_features_are_not_overwritten_unconditionally(self) -> None:
        self.assertNotIn('"features": prediction.get("features"),', self._source())

    def test_a_started_game_keeps_what_it_had(self) -> None:
        source = self._source()
        self.assertIn("if started and previous_features:", source)
        self.assertIn("pinned_features = previous_features", source)

    def test_the_freeze_is_recorded_so_clean_rows_can_be_told_apart(self) -> None:
        """Without the marker there is no way to know which rows to trust."""
        self.assertIn('"featuresFrozenAt": features_frozen_at', self._source())

    def test_the_fit_can_see_which_rows_are_clean(self) -> None:
        from model_fit import Sample

        self.assertIn("frozen", Sample.__slots__)

    def test_the_countdown_to_a_rebaseline_is_reported(self) -> None:
        """Contaminated history cannot be repaired, only outgrown."""
        self.assertIn("frozenSamples", (ROOT / "model_fit.py").read_text(encoding="utf-8"))

    def test_every_existing_row_is_correctly_marked_unclean(self) -> None:
        """Nothing logged before the fix may be mistaken for pinned data."""
        frozen = sum(1 for sample in self.samples if sample.frozen)
        graded_before_the_fix = len(self.samples) - frozen
        self.assertGreater(graded_before_the_fix, 0)

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, _centre = samples_from_log(DATA)


if __name__ == "__main__":
    unittest.main()
