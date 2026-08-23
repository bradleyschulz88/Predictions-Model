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
    """Scored against the real graded log, so it tracks the live data.

    Scored per POPULATION, not pooled, and that distinction is the whole point
    of this class. The log holds two kinds of row: those whose features were
    frozen at first pitch and are clean by construction, and those written
    before the freeze existed, which can carry the result. Pooling them
    averages a leak against clean data and hides it -- which is exactly what
    happened here on 2026-08-22, when the pooled `h2hDiff` score drifted under
    the ceiling and turned the fix working into a red build. Measured that day:

        h2hDiff  unfrozen 0.8515 (n=227)   frozen 0.6270 (n=194)   pooled 0.7494

    Both subgroup numbers are stable and meaningful. The pooled one is a
    mixture whose value depends only on how much clean data has accumulated
    since, so it says nothing about either group and drifts through the
    threshold on its own.

    That is the sixth time a pooled statistic has concealed contradicting
    subgroups in this project -- totals, spreads, the staking gate, closing
    line value, divergence, and now the leak detector itself. Written into the
    detector by the same hand that had just fixed the other five.
    """

    # Below this a subgroup AUC is noise rather than evidence.
    MIN_ROWS = 40

    @classmethod
    def setUpClass(cls) -> None:
        cls.samples, _centre = samples_from_log(DATA)
        cls.frozen = [s for s in cls.samples if getattr(s, "frozen", False)]
        cls.unfrozen = [s for s in cls.samples if not getattr(s, "frozen", False)]

    def _score(self, name: str, rows: list) -> tuple[float | None, int]:
        pairs = [
            (float(sample.values[name]), sample.label)
            for sample in rows
            if sample.values.get(name) is not None
        ]
        return (_auc(pairs) if len(pairs) >= self.MIN_ROWS else None), len(pairs)

    def _scored(self, name: str) -> tuple[float | None, int]:
        """Pooled. Kept for callers that genuinely want every row."""
        return self._score(name, self.samples)

    def _populations(self) -> tuple[tuple[str, list], ...]:
        return (("frozen", self.frozen), ("unfrozen", self.unfrozen))

    def test_the_market_is_where_it_should_be(self) -> None:
        """Calibrates the threshold. If this drifts, the threshold is wrong too.

        Checked in both populations. The market cannot leak in either -- a
        closing line is fixed before the game -- so a frozen/unfrozen gap here
        would mean the freeze itself is mislabelling rows rather than that any
        feature is contaminated.
        """
        for label, rows in self._populations():
            auc, count = self._score("marketLogit", rows)
            if auc is None:
                continue
            with self.subTest(population=label):
                self.assertGreater(auc, 0.55, "the closing line should carry real signal")
                self.assertLess(
                    auc, MAX_PLAUSIBLE_FEATURE_AUC, "the market itself cannot leak"
                )

    def test_no_shipped_or_candidate_feature_beats_the_plausible_ceiling(self) -> None:
        """Each population separately, so neither can dilute the other.

        A fresh leak shows up first in frozen rows, which are the minority --
        237 of 1143 at the time of writing. Pooled, it would be averaged
        against four times as many older rows and could sit under the ceiling
        for months before tripping.
        """
        offenders = []
        for name in CANDIDATE_FEATURES:
            for label, rows in self._populations():
                auc, count = self._score(name, rows)
                if auc is not None and auc > MAX_PLAUSIBLE_FEATURE_AUC:
                    offenders.append(f"{name} AUC {auc:.3f} on {count} {label} rows")
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

    def test_the_detector_catches_a_leak_it_is_shown(self) -> None:
        """Permanent proof, independent of what the log happens to contain.

        The original version of this asserted that the real `h2hDiff` still
        scored above the ceiling. That made the test a countdown: the leak was
        fixed, clean rows accumulated, the pooled score drifted from 0.855 to
        0.7494, and on 2026-08-22 it crossed 0.75 and failed every scheduled
        build. A test that fails when the bug is fixed is worse than no test,
        because the obvious response is to loosen the threshold that works.

        A synthetic leak cannot decay, so the detector's own behaviour is
        pinned here and the real data is left to say what it says below.
        """
        class _Row:
            def __init__(self, value: float, label: int) -> None:
                self.values, self.label = {"leaky": value}, label

        # A feature that simply reports the outcome, with enough noise that it
        # is not perfectly separable -- a real leak never is.
        rows = [_Row(0.9 if i % 5 else -0.4, 1) for i in range(60)]
        rows += [_Row(-0.9 if i % 5 else 0.4, 0) for i in range(60)]
        auc, count = self._score("leaky", rows)
        self.assertIsNotNone(auc, f"only {count} synthetic rows")
        self.assertGreater(
            auc, MAX_PLAUSIBLE_FEATURE_AUC,
            "the ceiling no longer catches a feature that reports the result",
        )

    def test_a_clean_feature_does_not_trip_the_detector(self) -> None:
        """The other half: a threshold that flags everything is not a detector.

        The market is the calibration point -- 0.645 on this log, the best
        honest pre-game estimator there is.
        """
        auc, count = self._score("marketLogit", self.samples)
        if auc is None:
            self.skipTest(f"only {count} rows carry a market price")
        self.assertLess(auc, MAX_PLAUSIBLE_FEATURE_AUC)

    def test_the_leak_is_still_visible_in_the_rows_that_carry_it(self) -> None:
        """Corroboration on real data, scored where the leak actually is.

        Unfrozen rows were written before features froze at first pitch, so
        they are the contaminated population and their score does not move as
        clean data accumulates: 0.8515 on 227 rows, against the 0.855 first
        measured. Skips rather than fails once those rows age out of the log,
        which is the correct end state -- by then the synthetic test above is
        the one holding the detector.
        """
        auc, count = self._score("h2hDiff", self.unfrozen)
        if auc is None:
            self.skipTest(f"only {count} pre-freeze rows still carry h2hDiff")
        self.assertGreater(
            auc, MAX_PLAUSIBLE_FEATURE_AUC,
            f"h2hDiff scored 0.855 when found and 0.8515 on {count} pre-freeze rows in "
            "Aug 2026. A drop here means these rows changed, which they should not",
        )

    def test_freezing_features_actually_cleaned_the_leak(self) -> None:
        """The claim the whole freeze was built on, measured rather than assumed.

        Same feature, same log, rows whose features were pinned at first pitch:
        0.6270 on 194 rows, against 0.8515 on the rows that were not. That gap
        is the fix working, and it is the number that was hidden inside the
        0.7494 pooled figure.
        """
        auc, count = self._score("h2hDiff", self.frozen)
        if auc is None:
            self.skipTest(f"only {count} frozen rows carry h2hDiff")
        self.assertLess(
            auc, MAX_PLAUSIBLE_FEATURE_AUC,
            f"h2hDiff scores {auc:.4f} on {count} rows frozen at first pitch. Those "
            "features are pinned before the game is played, so a leak here means the "
            "freeze is not holding",
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
