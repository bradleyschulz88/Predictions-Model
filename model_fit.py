"""Fit win-probability coefficients from graded outcomes (stdlib only).

The scoring model previously used hand-picked multipliers (record x3.2, splits
x2.4, power x2.2, form x1.8, ...) applied to features that measure largely the
same thing -- recordDiff and powerDiff correlate at 0.90 -- so one signal was
counted three or four times and the resulting probabilities were far too
confident. This module replaces the guesswork: it collapses the collinear
inputs into a single strength score and fits every coefficient by penalised
maximum likelihood against games that have actually been played.

Deliberately small: with a few hundred graded games per league, a handful of
regularised features is the most a fit can support without memorising noise.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from shared_utils import write_json

WEIGHTS_FILE = "model_weights.json"
ABLATION_FILE = "ablation.json"

# Feature order is part of the serialised model; append, never reorder.
#
# Chosen by walk-forward ablation (`python model_fit.py --ablate`), not by taste.
# Every additional feature measured worse out of sample on the 675 graded games
# available:
#
#   strengthDiff                    logloss 0.6443
#   + marketLogit                   logloss 0.6416   <- selected
#   + pitchingDiff                  logloss 0.6447
#   + restDiff                      logloss 0.6522
#   + injuryDiff                    logloss 0.6557
#   + b2bDiff                       logloss 0.6552
#
# That is a statement about these encodings at this sample size, not about the
# real world. Starting pitching genuinely matters in baseball -- but the market
# has already priced it, so once marketLogit is in the model, adding starter and
# bullpen ERA on top is redundant rather than additive.
#
# Five candidates carry an ASTERISK and are NOT fairly tested above:
#
#   restDiff, b2bDiff -- between 2026-07-23 and 2026-07-28 apply_predictions
#     re-ran enrichment per game without a schedule context, overwriting every
#     rest day and back-to-back flag with None/False. Roughly the last 200
#     graded games have no rest data at all, so these were scored on partly
#     destroyed input.
#
#   injurySeverityDiff -- added 2026-07-28 and present on zero graded games, so
#     it is imputed to the mean throughout and contributes exactly nothing. The
#     ablation showing it as neutral is measuring its absence, not the feature.
#
#   pitchingFipDiff -- FIP-based starter run-prevention, home minus away (see
#     _pitching_diff_fip). data_providers/mlb_pitcher.py was asking the MLB
#     Stats API for FIP under the "season" stat type, which cannot return it --
#     that field only exists under "sabermetrics" -- so this logged None on
#     every one of 872 graded games, the same always-absent signature already
#     caught for handedness/h2h/bullpen in test_matchup_context.py. Fixed to
#     query the right stat type; still starts from zero coverage, so it needs
#     the same weeks of fresh history as everything else on this list before
#     ablation can say anything real about it. _pitching_diff (ERA) already
#     loses once marketLogit is in the model (see above) -- the honest prior
#     is that FIP loses for the identical reason, since the market has priced
#     the starter too, but that is a hypothesis to test, not a promotion.
#
#   videoIntelDiff -- pre-game team news extracted from subscribed YouTube
#     channels (youtube_intel.py). Also present on zero graded games until the
#     local ingest has run for a while, so the same caveat applies.
#
#     The honest prior is that it will NOT help the priced leagues, for a
#     structural reason rather than a data-quality one: the model anchors to
#     marketLogit, and a de-vigged closing line already contains public
#     information. A preview show is public. By the time a pundit says it, it
#     is captioned and this scrapes it, the line has moved -- and information
#     already in the line contributes nothing by construction.
#
#     AFL is the case worth watching. It has no odds source, so STANDALONE_
#     FEATURES there is a single feature with no market to anchor to, which is
#     the one place extra signal has somewhere to go. n=46 is thin, so this
#     needs a real stretch of games before the answer means anything.
#
# eloDiff IS fairly tested and it loses. Unlike the others it can be computed
# retroactively, so it was measured properly on all 685 games:
#
#   strength + market            logloss 0.6443   <- kept
#   + eloDiff                    logloss 0.6478
#   eloDiff instead of strength  logloss 0.6656
#
# Elo knows about strength of schedule, which season records do not, and that
# should help. It does not yet, because ratings only start in mid-June: with 34
# MLB games at K=4 the updates add more noise than the seed already contains.
# Seeding from season record lifted its standalone correlation from +0.135 to
# +0.232, still short of recordDiff at +0.305. Worth re-testing after a full
# season, which is when Elo earns its reputation.
#
# All four need re-running once a couple of weeks of games have graded with
# the data actually present. Promote them here if the answer changes.
#
# The enrichment pipeline still supplies all of them to the reasoning panel;
# they are simply not allowed to move the probability until they can earn it.
ANCHORED_FEATURES = ("strengthDiff", "marketLogit")
STANDALONE_FEATURES = ("strengthDiff",)

# Every feature the collapser knows how to build, for ablation runs.
# Ordered by how much prior reason there is to believe them, because the
# ablation tests nested prefixes -- a feature is only ever judged alongside
# everything before it. The first two are what currently ships.
#
# Everything from h2hDiff onward was added later and has no coverage in the
# graded log yet, so today it contributes nothing and the ablation will
# correctly decline to ship it. That is the intended state: these are
# candidates accumulating evidence, not features waiting to be switched on.
#
# On sample size: there are ~700 graded games, and roughly 10-20 games per
# predictor is the honest limit, so this list is a queue to be tested one at a
# time rather than a model to be fitted all at once. Walk-forward is what
# enforces that -- nothing ships unless it beats its own absence out of sample.
CANDIDATE_FEATURES = (
    "strengthDiff",
    "marketLogit",
    "pitchingDiff",
    "pitchingFipDiff",
    "restDiff",
    "injuryDiff",
    "injurySeverityDiff",
    "eloDiff",
    "b2bDiff",
    "videoIntelDiff",
    "parkDiff",
    "travelDiff",
    "handednessDiff",
    "bullpenDiff",
    "formDiff",
)

# Features that carry the result of the game they are supposed to predict.
#
# `accuracy_tracker` used to overwrite a logged row's `features` on every build,
# and the build re-enriches dates that have already been played -- so anything
# read from a source that updates after a game encoded that game's outcome by
# the time it was scored. Features are now frozen at first pitch, so rows logged
# from 13 Aug 2026 carry `featuresFrozenAt` and are clean; every row before that
# is not, and no amount of later care recovers them.
#
# h2hDiff is the severe case and was caught the day ablate_each first ran. It
# reads ESPN's season-series summary ("Dodgers lead series 2-1"), and a series
# is three or four games, so one result moves it by a third. Measured 13 Aug
# over 268 graded rows it scored a standalone AUC of 0.855 and its sign agreed
# with the outcome on 79.9% of non-zero rows. For scale, the de-vigged closing
# line -- the best public pre-game estimator there is -- manages 0.640 on the
# same log, and the whole fitted model manages 0.665. No genuine head-to-head
# signal in baseball is worth 0.855.
#
# The nested ablation had been hiding this by accident: h2hDiff sat at position
# 11, behind eight features that had already degraded the fit, so it never got
# a chance to show what it was. Judged on its own it improves walk-forward log
# loss by 0.0370 -- five times the model's entire measured edge over the
# market, which is what a leak looks like rather than a discovery.
#
# Listed rather than deleted so the reason survives, and so a future candidate
# read from a post-game source has somewhere obvious to go.
LEAKING_FEATURES = ("h2hDiff",)

# Any candidate scoring above this standalone is not a pre-game signal. The
# market sits at 0.640 and is the ceiling for public information; a comfortable
# margin over it still leaves room for a genuinely strong feature without
# admitting anything that has seen the result.
# tests/test_feature_leakage.py enforces it.
MAX_PLAUSIBLE_FEATURE_AUC = 0.75

# Shrinkage constant for per-league intercepts: a league needs ~K graded games
# before its own home-field estimate outweighs the pooled one.
LEAGUE_INTERCEPT_PRIOR = 50.0

# Games a club needs before its record carries half the weight of the league
# mean in strengthDiff. Ten is about a sixth of a basketball season, a
# fortnight of baseball and most of an NFL year -- which is the right shape,
# because ten NFL games is genuinely more informative about a football team
# than ten baseball games are about a baseball one.
#
# Rows logged before 13 Aug 2026 carry no strengthGames and are left alone, so
# the fit sees them exactly as before rather than having a guess imputed for
# them. See build_feature_dict.
STRENGTH_SHRINK_GAMES = 10.0

# Probabilities are never published outside this band. Baseball in particular
# has no 95% games; the old 0.05/0.95 clamp let stacked features run to the rail.
MIN_PROB = 0.05
MAX_PROB = 0.95

# Fallback used only before a fit has measured the real value from the log.
DEFAULT_SPLIT_DIFF_CENTRE = 0.0457


# A probability that is not a number carries no information, so both maps
# below send it to the middle rather than to an end.
#
# That is not what they did. `logit` clamped with `max(1e-6, min(1 - 1e-6, p))`,
# and every comparison against NaN is False, so `min(0.999999, nan)` returns
# 0.999999 and the clamp came out at its CEILING. A NaN probability became
# logit 13.8, `apply_platt` returned 0.999999, and the board would have carried
# a 100.0% pick -- which also maximises the Kelly stake. Which end it landed on
# was decided by the order of the arguments to `min`: swap them and the same
# input clamps to the floor instead.
#
# Neither end is right. An input that means "no idea" must read as 0.5, which
# is below every publication threshold and stakes nothing.
NEUTRAL_PROBABILITY = 0.5


def sigmoid(value: float) -> float:
    if not math.isfinite(value):
        # An infinite logit is a real saturation and maps to the end it points
        # at; only NaN is the absence of an answer.
        if math.isnan(value):
            return NEUTRAL_PROBABILITY
        return 1.0 if value > 0 else 0.0
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def logit(prob: float) -> float:
    try:
        value = float(prob)
    except (TypeError, ValueError):
        value = NEUTRAL_PROBABILITY
    if not math.isfinite(value):
        value = NEUTRAL_PROBABILITY
    value = max(1e-6, min(1.0 - 1e-6, value))
    return math.log(value / (1.0 - value))


# --------------------------------------------------------------------------
# Linear algebra (small dense systems only)
# --------------------------------------------------------------------------


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot_row][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / pivot
            if factor:
                for col in range(column, size + 1):
                    augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = augmented[row][size] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, size)
        )
        solution[row] = total / augmented[row][row]
    return solution


# --------------------------------------------------------------------------
# Penalised logistic regression
# --------------------------------------------------------------------------


def fit_logistic(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    l2: float | Sequence[float] = 1.0,
    max_iterations: int = 50,
    tolerance: float = 1e-7,
) -> list[float]:
    """IRLS/Newton fit. Row vectors must already include a leading 1.0 intercept.

    The intercept is left unpenalised so regularisation shrinks effects toward
    zero without dragging the base rate away from the data.

    ``l2`` is either one ridge strength shared by every feature, or a sequence
    of per-feature strengths (length = width - 1, in row order, excluding the
    intercept). A shared scalar cannot express "trust this input more before
    seeing a single game" -- it shrinks every coefficient by the same
    proportion, which just preserves whatever split the unpenalised data
    implies. A per-feature penalty is a Gaussian prior with its own precision
    on each coefficient: tighter (more shrinkage) on a feature more likely to
    be noise, looser on one already known to be reliable.
    """
    if not rows:
        return []
    if len(rows) != len(labels):
        # `zip` below would silently truncate to the shorter of the two, which
        # means training on a subset while reporting the full count. A caller
        # that has lost the correspondence between a row and its outcome has a
        # bug worth stopping for.
        raise ValueError(f"{len(rows)} rows but {len(labels)} labels")

    # One non-finite value anywhere poisons the whole fit, not just its own
    # row: it enters the gradient and the Hessian, and every coefficient comes
    # back NaN. Dropping the row costs one observation. `_first_number` should
    # already have caught these upstream, so this is the backstop for a row
    # built by some other path.
    usable = [
        (row, label)
        for row, label in zip(rows, labels)
        if all(isinstance(x, (int, float)) and math.isfinite(x) for x in row)
        and label is not None
    ]
    if not usable:
        return []
    rows = [row for row, _ in usable]
    labels = [label for _, label in usable]

    width = len(rows[0])
    if isinstance(l2, (int, float)):
        penalties = [0.0] + [float(l2)] * (width - 1)
    else:
        penalties = [0.0] + [float(value) for value in l2]
        if len(penalties) != width:
            raise ValueError(
                f"l2 has {len(penalties) - 1} per-feature entries but the rows have "
                f"{width - 1} features"
            )
    weights = [0.0] * width

    for _ in range(max_iterations):
        gradient = [0.0] * width
        hessian = [[0.0] * width for _ in range(width)]

        for row, label in zip(rows, labels):
            prediction = sigmoid(sum(w * x for w, x in zip(weights, row)))
            residual = label - prediction
            # Floor the weight so a saturated fit cannot produce a singular Hessian.
            variance = max(prediction * (1.0 - prediction), 1e-8)
            for i in range(width):
                gradient[i] += residual * row[i]
                row_i_var = row[i] * variance
                for j in range(width):
                    hessian[i][j] += row_i_var * row[j]

        for i in range(width):
            penalty = penalties[i]
            gradient[i] -= penalty * weights[i]
            hessian[i][i] += penalty

        step = solve_linear(hessian, gradient)
        if step is None:
            break
        weights = [w + s for w, s in zip(weights, step)]
        if max(abs(s) for s in step) < tolerance:
            break

    return weights


def predict_row(weights: Sequence[float], row: Sequence[float]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, row)))


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def _first_number(*values: Any) -> float | None:
    """The first value that converts to a real number, else None.

    Non-finite floats are treated as absent rather than passed through.
    `float("nan")` converts without complaint, and one NaN reaching the fit is
    not a degraded row -- IRLS propagates it through the gradient and the
    Hessian, so every coefficient comes back NaN and the model stops making
    predictions at all. Measured: ten NaN rows among twenty returned
    `[nan, nan]` for a two-weight fit.

    None is a shape the pipeline already knows how to handle: it means the
    feature was not available for this game, which is exactly what an
    unusable number amounts to.
    """
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def build_feature_dict(
    features: dict[str, Any] | None,
    *,
    split_diff_centre: float | None = None,
) -> dict[str, float | None]:
    """Collapse the logged feature blob into the small set the model fits on.

    recordDiff, splitDiff and powerDiff correlate at 0.82-0.90 -- they are three
    measurements of team quality, not three signals. Averaging whichever are
    present yields one input, so the fit assigns team strength a single
    coefficient instead of stacking three.
    """
    features = features or {}
    centre = DEFAULT_SPLIT_DIFF_CENTRE if split_diff_centre is None else split_diff_centre

    strength_parts: list[float] = []
    record_diff = _first_number(features.get("recordDiff"))
    if record_diff is not None:
        strength_parts.append(record_diff)
    split_diff = _first_number(features.get("splitDiff"))
    if split_diff is not None:
        # Home/road splits embed home-field advantage. Centring on the observed
        # league-average gap leaves only team-specific home strength, so the
        # fitted intercept carries home field exactly once.
        strength_parts.append(split_diff - centre)
    home_power = _first_number(features.get("homePower"))
    away_power = _first_number(features.get("awayPower"))
    if home_power is not None and away_power is not None:
        strength_parts.append(home_power - away_power)

    strength = sum(strength_parts) / len(strength_parts) if strength_parts else None

    # Shrink toward the league mean by how many games the estimate rests on.
    #
    # A 1-0 record and a 12-4 record both produced a win percentage and nothing
    # told them apart, so one result was read with the authority of a third of
    # a season. It was not hypothetical: on 13 Aug 2026 NFL preseason picks ran
    # a median 27.2 points from the market with 70% over 15 -- against 8.2 and
    # 25% for baseball -- because a 0.80 to 0.00 power gap off single opening
    # games dominated a market anchor fitted on mature records. Three picks sat
    # at 87-95% confidence where the market said 41-45%, one of them pinned to
    # the MAX_PROB clamp.
    #
    # `count / (count + k)` is the same shrinkage already used for per-league
    # intercepts, and the same reason: an estimate earns its weight from its
    # sample. At one game and k=10 a club's own record carries 9%; by twenty
    # games it carries two thirds; by a full season it is essentially untouched,
    # which is why this is close to a no-op on the leagues that already work.
    #
    # It is deliberately general rather than an NFL rule. The same defect
    # arrives in MLB every April and the NBA every October, and a league gate
    # would have to be remembered each time. Preseason reports zero games, so
    # exhibitions fall out of the same arithmetic with no second code path.
    if strength is not None:
        games = _first_number(features.get("strengthGames"))
        if games is not None:
            strength *= games / (games + STRENGTH_SHRINK_GAMES)

    implied_home = _first_number(features.get("impliedHome"))
    market_logit = logit(implied_home / 100.0) if implied_home is not None else None

    home_rest = _first_number(features.get("homeRest"))
    away_rest = _first_number(features.get("awayRest"))
    rest_diff = (home_rest - away_rest) if home_rest is not None and away_rest is not None else None

    home_injury = _first_number(features.get("homeInjuryLoad")) or 0.0
    away_injury = _first_number(features.get("awayInjuryLoad")) or 0.0

    # Severity-weighted alternative to the raw count. None until a game carries
    # it, so older logged games contribute nothing rather than a false zero.
    home_severity = _first_number(features.get("homeInjurySeverity"))
    away_severity = _first_number(features.get("awayInjurySeverity"))
    severity_diff = (
        away_severity - home_severity
        if home_severity is not None and away_severity is not None
        else None
    )

    # Pre-game Elo rating gap, in rating points scaled to a comparable range.
    elo_edge = _first_number(features.get("eloEdge"))
    elo_diff = elo_edge / 100.0 if elo_edge is not None else None

    # Pre-game team news scraped from subscribed YouTube channels, home minus
    # away, already leakage-guarded on the video's publish time by
    # youtube_intel.intel_edge. None whenever either side went uncovered.
    video_intel = _first_number(features.get("videoIntelEdge"))

    home_b2b = 1.0 if features.get("homeBackToBack") else 0.0
    away_b2b = 1.0 if features.get("awayBackToBack") else 0.0

    return {
        "strengthDiff": strength,
        "marketLogit": market_logit,
        "pitchingDiff": _pitching_diff(features),
        "pitchingFipDiff": _pitching_diff_fip(features),
        "restDiff": rest_diff,
        "injuryDiff": away_injury - home_injury,
        "injurySeverityDiff": severity_diff,
        "eloDiff": elo_diff,
        "b2bDiff": away_b2b - home_b2b,
        "videoIntelDiff": video_intel,
        # Newest candidates. These have no history in the log yet, so they are
        # all-None for every graded game today and contribute exactly zero to
        # the fit -- an absent feature is standardised to the mean, not to an
        # extreme. They start earning coverage from the build that ships them
        # and cannot be judged until a few weeks of games have graded with the
        # data present.
        "h2hDiff": _first_number(features.get("h2hDiff")),
        "parkDiff": _first_number(features.get("parkEdge")),
        "travelDiff": _first_number(features.get("travelDiff")),
        "handednessDiff": _first_number(features.get("handednessDiff")),
        "bullpenDiff": _first_number(features.get("bullpenDiff")),
        # Last-five win rate, home minus away. The live log folds form into
        # strengthDiff rather than carrying it separately, so this is only
        # populated by the backfill -- which is exactly the point: recent form is
        # cheap to reconstruct historically and is worth a fair test.
        "formDiff": _first_number(features.get("formDiff")),
    }


# The starter throws roughly two thirds of a game and the bullpen the rest, so
# the run-prevention edge is weighted accordingly. Sign convention matches the
# rest: positive favours the home side.
STARTER_WEIGHT = 0.65
BULLPEN_WEIGHT = 0.35


def _pitching_diff(features: dict[str, Any]) -> float | None:
    """Run-prevention edge in ERA units, home minus away.

    MLB-only, and None elsewhere so it contributes nothing to other leagues.
    Uses ERA rather than FIP because starter and bullpen ERA are present on
    97-100% of games, where FIP -- see _pitching_diff_fip -- is only just
    starting to be logged at all.
    """
    pitching = features.get("mlbPitching")
    if not isinstance(pitching, dict):
        return None

    parts: list[tuple[float, float]] = []

    home_starter = _first_number(
        pitching.get("homePitcherRecentEra"), pitching.get("homePitcherApiEra")
    )
    away_starter = _first_number(
        pitching.get("awayPitcherRecentEra"), pitching.get("awayPitcherApiEra")
    )
    if home_starter is not None and away_starter is not None:
        parts.append((away_starter - home_starter, STARTER_WEIGHT))

    home_bullpen = _first_number(pitching.get("homeBullpenEra"))
    away_bullpen = _first_number(pitching.get("awayBullpenEra"))
    if home_bullpen is not None and away_bullpen is not None:
        parts.append((away_bullpen - home_bullpen, BULLPEN_WEIGHT))

    if not parts:
        return None
    weight_total = sum(weight for _, weight in parts)
    return sum(value * weight for value, weight in parts) / weight_total


def _pitching_diff_fip(features: dict[str, Any]) -> float | None:
    """Starter run-prevention edge in FIP units, home minus away.

    FIP (fielding-independent pitching) prices only strikeouts, walks and
    home runs -- the outcomes a pitcher controls without their defense behind
    them -- where ERA also carries whatever that day's fielders and luck did.
    It is the standard sabermetric alternative for exactly that reason.

    Starters only: team FIP is not fetched for the bullpen, so this does not
    blend in an ERA-only relief term the way _pitching_diff does. A separate
    candidate rather than a replacement for it -- data_providers/mlb_pitcher.py
    only just started actually retrieving FIP (it was querying a stat type
    that cannot return it, silently, on every graded game so far), so this
    has no real coverage yet and CANDIDATE_FEATURES documents it accordingly.
    Judge it against its own absence once some weeks of games have graded
    with the data actually present, the same way every other queued
    candidate here is judged.
    """
    pitching = features.get("mlbPitching")
    if not isinstance(pitching, dict):
        return None
    home_fip = _first_number(pitching.get("homePitcherFip"))
    away_fip = _first_number(pitching.get("awayPitcherFip"))
    if home_fip is None or away_fip is None:
        return None
    return away_fip - home_fip


def measure_split_diff_centre(features_list: Sequence[dict[str, Any]]) -> float:
    """Average home-record minus away-record gap across the training set.

    This is the portion of splitDiff that is simply home-field advantage rather
    than team quality, and it is what gets subtracted so the fitted intercept
    carries home field exactly once instead of twice.
    """
    values = [
        float(features["splitDiff"])
        for features in features_list
        if (features or {}).get("splitDiff") is not None
    ]
    return round(sum(values) / len(values), 4) if values else DEFAULT_SPLIT_DIFF_CENTRE


def to_row(
    values: dict[str, float | None],
    feature_names: Sequence[str],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[float]:
    """Standardised design row with a leading intercept.

    Missing values fall back to the training mean, which after centring is zero
    -- an absent feature therefore contributes nothing rather than silently
    reading as an extreme value.
    """
    row = [1.0]
    for name in feature_names:
        value = values.get(name)
        if value is None:
            row.append(0.0)
            continue
        scale = scales.get(name) or 1.0
        row.append((value - means.get(name, 0.0)) / scale)
    return row


def standardisation(
    samples: Sequence[dict[str, float | None]], feature_names: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        present = [row[name] for row in samples if row.get(name) is not None]
        if not present:
            means[name], scales[name] = 0.0, 1.0
            continue
        mean = sum(present) / len(present)
        variance = sum((value - mean) ** 2 for value in present) / len(present)
        means[name] = mean
        scales[name] = math.sqrt(variance) or 1.0
    return means, scales


# --------------------------------------------------------------------------
# Model object
# --------------------------------------------------------------------------


class ProbabilityModel(Protocol):
    """Seam for swapping in a stronger learner without touching the caller.

    A gradient-boosted model lived in ml_model/ for a while and was deleted: it
    was never on the prediction path, needed numpy, scikit-learn and xgboost
    against a stdlib-only runtime, and its headline metrics (log loss 0.582,
    Brier 0.206, AUC 0.738) were not comparable to the live model's -- different
    sample, synthetic rows included, per-fold AUC ranging 0.47 to 0.66 at n=510.

    Anything dropped in here later has to earn its place the same way every
    current feature did, whatever its own training metrics say:

        python model_fit.py --ablate
        python scripts/check_regression.py

    both compare against the live walk-forward baseline in
    docs/data/model_baseline.json. A model that cannot beat that out of sample
    should not ship.
    """

    def predict_proba(self, features: dict[str, Any], league: str) -> float | None:
        ...


class LogisticModel:
    """Fitted logistic model with per-league intercepts."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self._anchored = payload.get("anchored") or {}
        self._standalone = payload.get("standalone") or {}
        self._league_intercepts = payload.get("leagueIntercepts") or {}
        self.split_diff_centre = float(
            payload.get("splitDiffCentre", DEFAULT_SPLIT_DIFF_CENTRE)
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._payload.get("metadata") or {}

    def predict_proba(self, features: dict[str, Any], league: str) -> float | None:
        """Home win probability from a raw logged/live feature blob."""
        values = build_feature_dict(features, split_diff_centre=self.split_diff_centre)
        return self.predict_from_values(values, league)

    def _block_for(self, values: dict[str, float | None]) -> dict[str, Any] | None:
        use_anchored = values.get("marketLogit") is not None and self._anchored.get("weights")
        block = self._anchored if use_anchored else self._standalone
        return block if block.get("weights") else None

    def predict_from_values(
        self, values: dict[str, float | None], league: str
    ) -> float | None:
        """Home win probability from an already-collapsed feature dict."""
        block = self._block_for(values)
        if block is None:
            return None

        names = block.get("features") or []
        row = to_row(values, names, block.get("means") or {}, block.get("scales") or {})
        score = sum(w * x for w, x in zip(block["weights"], row))
        score += float(self._league_intercepts.get(league, 0.0))
        return max(MIN_PROB, min(MAX_PROB, sigmoid(score)))

    def explain(
        self, values: dict[str, float | None], league: str
    ) -> list[dict[str, Any]] | None:
        """Per-feature logit contributions behind a prediction.

        The published explanation used to describe whatever enrichment happened
        to be available, which drifted away from what the model actually reads.
        This returns the real decomposition so the two cannot disagree.
        """
        block = self._block_for(values)
        if block is None:
            return None

        names = block.get("features") or []
        weights = block["weights"]
        row = to_row(values, names, block.get("means") or {}, block.get("scales") or {})

        contributions: list[dict[str, Any]] = [
            {
                "feature": "homeField",
                # Intercept plus the league's own correction: the baseline edge
                # a home side gets before any team-specific information.
                "contribution": round(weights[0] + float(self._league_intercepts.get(league, 0.0)), 4),
                "value": None,
                "available": True,
            }
        ]
        for name, weight, standardised in zip(names, weights[1:], row[1:]):
            contributions.append(
                {
                    "feature": name,
                    "contribution": round(weight * standardised, 4),
                    "value": values.get(name),
                    # A feature imputed to the training mean contributes nothing
                    # and should not be presented as a reason either way.
                    "available": values.get(name) is not None,
                }
            )
        return contributions

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def load_model(data_dir: Path | None = None) -> LogisticModel | None:
    """Load fitted weights, returning None so callers can fall back cleanly."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "docs" / "data"
    path = data_dir / WEIGHTS_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not payload.get("anchored") and not payload.get("standalone"):
        return None
    if not _blocks_are_consistent(payload):
        return None
    return LogisticModel(payload)


def _blocks_are_consistent(payload: dict[str, Any]) -> bool:
    """Every block's weights must line up with its own feature list.

    predict_from_values scores with `zip(weights, row)`, and zip stops at the
    shorter side. A weights list one item short therefore drops the last
    feature and returns a confident, wrong number rather than raising: on the
    live file, truncating one weight moved a published probability from 76.3%
    to 65.5% with nothing logged. This file is regenerated every build and is
    gitignored, so a partial write has no reviewer and no diff to notice it.

    Refusing the file falls back to the heuristic path, which the build already
    treats as a first-class outcome, and the refit gate fails the build when
    model_fit.py produces nothing usable. Both are loud. A silently truncated
    model is not.
    """
    for name in ("anchored", "standalone"):
        block = payload.get(name)
        if not block:
            continue
        weights = block.get("weights")
        features = block.get("features")
        if not isinstance(weights, list) or not isinstance(features, list):
            return False
        # One weight per feature, plus the leading intercept that to_row emits.
        if len(weights) != len(features) + 1:
            return False
        if not all(isinstance(w, (int, float)) and math.isfinite(w) for w in weights):
            return False
    return True


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


class Sample:
    __slots__ = ("values", "label", "league", "date", "has_market", "frozen")

    def __init__(
        self,
        values: dict[str, float | None],
        label: int,
        league: str,
        date: str,
        frozen: bool = False,
    ) -> None:
        self.values = values
        self.label = label
        self.league = league
        self.date = date
        self.has_market = values.get("marketLogit") is not None
        # Whether the feature vector was pinned before the game started. False
        # means it was recomputed afterwards and may carry the result -- see
        # the freeze comment in accuracy_tracker. Every row logged before
        # 13 Aug 2026 is False, because nothing was pinning them.
        self.frozen = frozen


def load_history_samples(data_dir: Path) -> list[tuple[dict[str, Any], int, str, str]]:
    """Backfilled pre-game rows from completed seasons, if any have been built.

    Optional by design: `scripts/backfill_history.py` writes this on demand and
    it is gitignored, so a fresh clone simply has none and the fit behaves
    exactly as before.

    These rows carry NO market feature, because historical closing lines are not
    available. That makes them a screening set for the standalone model rather
    than extra training data for the live one -- mixing them into the anchored
    fit would let thousands of market-less games swamp the few hundred that
    actually have a price.
    """
    history = _load_json(data_dir / "history_features.json", {"rows": []})
    rows: list[tuple[dict[str, Any], int, str, str]] = []
    for row in history.get("rows") or []:
        features = row.get("features") or {}
        label = row.get("homeWon")
        if label is None:
            continue
        rows.append((features, int(label), row.get("league") or "unknown", row.get("date") or ""))
    return rows


def samples_from_log(data_dir: Path) -> tuple[list[Sample], float]:
    """Join logged features to graded outcomes.

    Returns the samples plus the split-diff centre measured from this data, so
    the same value is used at fit time and at prediction time.
    """
    log = _load_json(data_dir / "predictions_log.json", {"predictions": {}})
    accuracy = _load_json(data_dir / "accuracy.json", {"picksByEventId": {}})

    raw: list[tuple[dict[str, Any], int, str, str]] = []
    for event_id, graded in (accuracy.get("picksByEventId") or {}).items():
        if graded.get("status") != "graded":
            continue
        logged = (log.get("predictions") or {}).get(event_id) or {}
        merged = {**logged, **graded}
        label = _home_label(merged)
        if label is None:
            continue
        raw.append(
            (
                merged.get("features") or {},
                label,
                merged.get("league") or "unknown",
                merged.get("date") or merged.get("scheduleDate") or "",
                bool(merged.get("featuresFrozenAt")),
            )
        )

    centre = measure_split_diff_centre([features for features, _, _, _, _ in raw])
    samples = [
        Sample(
            values=build_feature_dict(features, split_diff_centre=centre),
            label=label,
            league=league,
            date=date,
            frozen=frozen,
        )
        for features, label, league, date, frozen in raw
    ]
    return samples, centre


def _home_label(record: dict[str, Any]) -> int | None:
    home_score, away_score = record.get("homeScore"), record.get("awayScore")
    if home_score is None or away_score is None:
        return None
    try:
        home, away = int(home_score), int(away_score)
    except (TypeError, ValueError):
        return None
    if home == away:
        return None  # Draws are handled separately, not as a home/away label.
    return 1 if home > away else 0


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _l2_vector(l2: float | dict[str, float], feature_names: Sequence[str]) -> float | list[float]:
    """Resolve a per-feature penalty dict into fit_logistic's expected vector.

    A dict scopes naturally to whichever block is being fit: the standalone
    block only has strengthDiff, so a dict built for the anchored pair still
    resolves correctly there -- marketLogit's entry is simply unused. Any
    feature not named in the dict falls back to "_default", so adding a new
    candidate feature later does not require touching every caller.
    """
    if isinstance(l2, (int, float)):
        return l2
    default = l2.get("_default", 1.0)
    return [l2.get(name, default) for name in feature_names]


def _fit_block(
    samples: Sequence[Sample], feature_names: Sequence[str], *, l2: float | dict[str, float]
) -> dict[str, Any] | None:
    if len(samples) < len(feature_names) * 5:
        return None
    values = [sample.values for sample in samples]
    means, scales = standardisation(values, feature_names)
    rows = [to_row(sample.values, feature_names, means, scales) for sample in samples]
    labels = [sample.label for sample in samples]
    weights = fit_logistic(rows, labels, l2=_l2_vector(l2, feature_names))
    if not weights:
        return None
    return {
        "features": list(feature_names),
        "weights": weights,
        "means": means,
        "scales": scales,
        "n": len(samples),
        "l2": l2,
    }


def _league_intercepts(
    samples: Sequence[Sample], anchored: dict[str, Any] | None, standalone: dict[str, Any] | None
) -> dict[str, float]:
    """Per-league home-field correction, shrunk toward the pooled fit.

    A league with 46 graded games should barely move its own intercept; one with
    478 should move it most of the way.
    """
    by_league: dict[str, list[Sample]] = {}
    for sample in samples:
        by_league.setdefault(sample.league, []).append(sample)

    intercepts: dict[str, float] = {}
    for league, league_samples in by_league.items():
        residual_sum = 0.0
        count = 0
        for sample in league_samples:
            block = anchored if (sample.has_market and anchored) else standalone
            if not block:
                continue
            row = to_row(sample.values, block["features"], block["means"], block["scales"])
            predicted = predict_row(block["weights"], row)
            residual_sum += sample.label - predicted
            count += 1
        if not count:
            continue
        # One Newton step on the intercept alone, then shrink toward zero.
        mean_residual = residual_sum / count
        raw = mean_residual / 0.25  # 0.25 is the max of p(1-p); a conservative step.
        shrink = count / (count + LEAGUE_INTERCEPT_PRIOR)
        intercepts[league] = round(raw * shrink, 4)
    return intercepts


def fit_from_observations(
    samples: Sequence[Sample],
    *,
    l2: float | dict[str, float] = 1.0,
    split_diff_centre: float = DEFAULT_SPLIT_DIFF_CENTRE,
    anchored_features: Sequence[str] = ANCHORED_FEATURES,
    standalone_features: Sequence[str] = STANDALONE_FEATURES,
) -> dict[str, Any]:
    """Fit anchored and standalone blocks plus per-league intercepts."""
    anchored_samples = [sample for sample in samples if sample.has_market]
    anchored = _fit_block(anchored_samples, anchored_features, l2=l2)
    standalone = _fit_block(samples, standalone_features, l2=l2)
    intercepts = _league_intercepts(samples, anchored, standalone)

    return {
        "anchored": anchored or {},
        "standalone": standalone or {},
        "leagueIntercepts": intercepts,
        "splitDiffCentre": split_diff_centre,
        "metadata": {
            "nTotal": len(samples),
            "nWithMarket": len(anchored_samples),
            "l2": l2,
            "leagues": sorted({sample.league for sample in samples}),
        },
    }


def walk_forward_scores(
    samples: Sequence[Sample],
    *,
    l2: float | dict[str, float] = 1.0,
    folds: int = 5,
    anchored_features: Sequence[str] = ANCHORED_FEATURES,
    standalone_features: Sequence[str] = STANDALONE_FEATURES,
) -> dict[str, Any]:
    """Expanding-window evaluation: never score a game the fit has seen.

    Sports data is a time series and the model is retrained on its own history,
    so a random split would leak future information and flatter the model.
    """
    ordered = sorted(samples, key=lambda sample: (sample.date, sample.league))
    if len(ordered) < folds * 20:
        return {"folds": 0, "note": "insufficient data for walk-forward"}

    start = len(ordered) // (folds + 1)
    predictions: list[tuple[float, int]] = []
    for fold in range(folds):
        split = start * (fold + 1)
        train, test = ordered[:split], ordered[split : split + start]
        if not test:
            continue
        payload = fit_from_observations(
            train,
            l2=l2,
            anchored_features=anchored_features,
            standalone_features=standalone_features,
        )
        model = LogisticModel(payload)
        for sample in test:
            # Values are already collapsed; re-deriving them would re-centre
            # against a different training set and leak information.
            prob = model.predict_from_values(sample.values, sample.league)
            if prob is not None:
                # Keep the market's own view of the same game alongside the
                # model's. Scoring them on identical games out of sample is the
                # only fair comparison; the published record pools every model
                # version this log has carried and describes history instead.
                predictions.append(
                    (prob, sample.label, sample.values.get("marketLogit"), sample.league)
                )

    if not predictions:
        return {"folds": 0}

    log_loss = -sum(
        math.log(max(1e-9, prob if label else 1.0 - prob)) for prob, label, _, _ in predictions
    ) / len(predictions)
    brier = sum((prob - label) ** 2 for prob, label, _, _ in predictions) / len(predictions)
    hits = sum(1 for prob, label, _, _ in predictions if (prob >= 0.5) == bool(label))

    scores = {
        "folds": folds,
        "n": len(predictions),
        "logLoss": round(log_loss, 4),
        "brier": round(brier, 4),
        "accuracy": round(hits / len(predictions), 4),
    }

    # Head-to-head against the market on exactly the games where a price
    # existed. Without this the only market comparison available was against
    # `model (published)`, which pools every model version the log has ever
    # carried -- so a fixed model still reads as losing to the market for as
    # long as its own bad history dominates the record.
    priced = [
        (prob, label, market)
        for prob, label, market, _ in predictions
        if market not in (None, 0.0)
    ]
    if priced:
        model_loss = -sum(
            math.log(max(1e-9, prob if label else 1.0 - prob)) for prob, label, _ in priced
        ) / len(priced)
        market_loss = -sum(
            math.log(max(1e-9, sigmoid(market) if label else 1.0 - sigmoid(market)))
            for _, label, market in priced
        ) / len(priced)
        scores["vsMarket"] = {
            "n": len(priced),
            "modelLogLoss": round(model_loss, 4),
            "marketLogLoss": round(market_loss, 4),
            "edge": round(market_loss - model_loss, 4),
        }

    # Home bias for the model that is running now. The published figure is
    # computed on whatever version made each pick, and it read MLB at +6.4pts
    # -- a large, specific-looking fault that the live model does not have
    # (+0.4pts). Reported with its binomial standard error, because on a
    # league with 95 graded games a 4pt gap is one standard error and means
    # nothing on its own.
    home_bias: dict[str, Any] = {}
    leagues = sorted({league for _, _, _, league in predictions})
    for league in leagues:
        pool = [(prob, label) for prob, label, _, lg in predictions if lg == league]
        if not pool:
            continue
        picked_home = sum(1 for prob, _ in pool if prob > 0.5) / len(pool) * 100
        actual_home = sum(label for _, label in pool) / len(pool) * 100
        std_err = math.sqrt(0.25 / len(pool)) * 100
        home_bias[league] = {
            "n": len(pool),
            "pickHomePct": round(picked_home, 1),
            "actualHomeWinPct": round(actual_home, 1),
            "biasPct": round(picked_home - actual_home, 1),
            "stdErrPct": round(std_err, 1),
            "significant": abs(picked_home - actual_home) > 1.96 * std_err,
        }
    if home_bias:
        scores["homeBias"] = home_bias
    return scores


def choose_l2(
    samples: Sequence[Sample],
    candidates: Iterable[float] = (0.3, 1.0, 3.0, 10.0, 30.0),
    **feature_kwargs: Any,
) -> float:
    """Pick the ridge strength with the best walk-forward log loss."""
    best_l2, best_loss = 1.0, float("inf")
    for candidate in candidates:
        scores = walk_forward_scores(samples, l2=candidate, **feature_kwargs)
        loss = scores.get("logLoss")
        if loss is not None and loss < best_loss:
            best_l2, best_loss = candidate, loss
    return best_l2


# Candidate ridge strengths either side of the shared-scalar search above.
# strengthDiff's own candidates run wider and heavier: it is an average of
# three season-aggregate stats standing in for a single "true" quality signal,
# so more shrinkage toward zero is the reasonable prior. marketLogit's run
# lighter: a de-vigged closing price is already close to the best available
# estimate, so the fit should have to work to justify pulling it toward zero
# at all.
STRENGTH_L2_CANDIDATES: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)
MARKET_L2_CANDIDATES: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


def choose_anchored_penalties(
    samples: Sequence[Sample],
    *,
    strength_candidates: Iterable[float] = STRENGTH_L2_CANDIDATES,
    market_candidates: Iterable[float] = MARKET_L2_CANDIDATES,
    default_l2: float = 10.0,
) -> dict[str, float]:
    """Pick separate ridge strengths for strengthDiff and marketLogit.

    choose_l2 shares one penalty across every feature, which shrinks every
    coefficient by the same proportion and therefore preserves whatever split
    the unpenalised data implies -- on this project's graded history that
    split lands close to even (51.7% strengthDiff / 48.3% marketLogit by
    relative weight) despite the de-vigged market alone beating the blended
    model's log loss on every game both have priced. A shared scalar has no
    way to express "trust one of these more before seeing a single game";
    that has to be an asymmetric prior, which unequal per-feature ridge
    strength is -- a Gaussian prior on each coefficient with its own
    precision, centred at zero, tighter on the input more likely to be noise.

    Still chosen by walk-forward, not by asserting a ratio: the candidate
    grids above encode the prior that strengthDiff should generally be
    shrunk harder, but the actual pair is whichever one measures best out of
    sample, the same discipline choose_l2 already applies to a single
    scalar.
    """
    best = {"strengthDiff": default_l2, "marketLogit": default_l2}
    best_loss = float("inf")
    for strength_l2 in strength_candidates:
        for market_l2 in market_candidates:
            l2_map = {
                "strengthDiff": strength_l2,
                "marketLogit": market_l2,
                "_default": default_l2,
            }
            scores = walk_forward_scores(
                samples,
                l2=l2_map,
                anchored_features=ANCHORED_FEATURES,
                standalone_features=STANDALONE_FEATURES,
            )
            loss = scores.get("logLoss")
            if loss is not None and loss < best_loss:
                best_loss = loss
                best = {"strengthDiff": strength_l2, "marketLogit": market_l2}
    return best


def _best_over_ridge(
    samples: Sequence[Sample], features: Sequence[str]
) -> dict[str, Any] | None:
    """Walk-forward score for one feature set, at its best ridge."""
    anchored = tuple(features)
    standalone = tuple(name for name in anchored if name != "marketLogit")
    best: dict[str, Any] | None = None
    for l2 in (1.0, 3.0, 10.0, 30.0):
        scores = walk_forward_scores(
            samples,
            l2=l2,
            anchored_features=anchored,
            standalone_features=standalone,
        )
        if scores.get("logLoss") is None:
            continue
        if best is None or scores["logLoss"] < best["logLoss"]:
            best = {**scores, "l2": l2}
    return best


def ablate(samples: Sequence[Sample]) -> list[dict[str, Any]]:
    """Walk-forward score of each nested feature set, best ridge per set.

    Keeps feature selection honest and repeatable: as the graded log grows, a
    feature that cannot beat its own absence should not be in the model.

    Nested prefixes answer one question and are routinely read as answering a
    different one -- see `ablate_each` below, which supplies the other two.
    """
    results: list[dict[str, Any]] = []
    for size in range(1, len(CANDIDATE_FEATURES) + 1):
        best = _best_over_ridge(samples, CANDIDATE_FEATURES[:size])
        if best:
            results.append({"features": list(CANDIDATE_FEATURES[:size]), **best})
    return results


def ablate_each(samples: Sequence[Sample]) -> dict[str, list[dict[str, Any]]]:
    """Judge every candidate on its own merits, twice.

    The nested sweep above tests prefixes of a fixed, prior-ordered list, so a
    candidate at position five is only ever scored with the four ahead of it
    already in the model, and once an early feature hurts, everything after it
    inherits the damage. Measured 12 Aug the shape was exactly that: log loss
    improved to 0.6401 at position three and then degraded monotonically for
    all thirteen features behind it.

    Read casually, that says "no candidate helps". What it actually licenses is
    "no candidate helps when stacked behind everything ahead of it in an
    arbitrary order", which is a much weaker claim and not the one feature
    selection needs. A genuinely useful feature sitting at position nine cannot
    demonstrate anything from there.

    Two sweeps fix it, both reusing the same walk-forward machinery:

    * **added** -- the shipped set plus one candidate, each independently. This
      is the question actually being asked: does this feature earn a place
      next to what already ships?
    * **removed** -- the full candidate set minus one, each independently. This
      catches the opposite error, a feature that only looks useless because
      another one is standing in for it.

    Both are scored against the shipped baseline, so a row's `delta` is the
    change in walk-forward log loss it causes. Negative is an improvement.
    """
    shipped = _best_over_ridge(samples, ANCHORED_FEATURES)
    baseline = shipped["logLoss"] if shipped else None

    added: list[dict[str, Any]] = []
    for name in CANDIDATE_FEATURES:
        if name in ANCHORED_FEATURES:
            continue
        scored = _best_over_ridge(samples, (*ANCHORED_FEATURES, name))
        if scored:
            added.append({
                "feature": name,
                "delta": None if baseline is None else round(scored["logLoss"] - baseline, 5),
                **scored,
            })
    added.sort(key=lambda row: (row["delta"] is None, row["delta"]))

    full = _best_over_ridge(samples, CANDIDATE_FEATURES)
    full_loss = full["logLoss"] if full else None
    removed: list[dict[str, Any]] = []
    for name in CANDIDATE_FEATURES:
        if name in ANCHORED_FEATURES:
            continue
        remaining = tuple(f for f in CANDIDATE_FEATURES if f != name)
        scored = _best_over_ridge(samples, remaining)
        if scored:
            # Positive means dropping it made things worse, i.e. it was pulling
            # its weight inside the full set.
            removed.append({
                "feature": name,
                "delta": None if full_loss is None else round(scored["logLoss"] - full_loss, 5),
                **scored,
            })
    removed.sort(key=lambda row: (row["delta"] is None, -(row["delta"] or 0.0)))

    return {
        "shippedBaseline": shipped,
        "fullSet": full,
        "added": added,
        "removed": removed,
    }


def ablate_and_write(
    data_dir: Path, samples: Sequence[Sample] | None = None
) -> dict[str, Any]:
    """Run the walk-forward ablation and persist it next to the other reports.

    The comment above ANCHORED_FEATURES already says every queued candidate
    (h2h, handedness, bullpen, elo, ...) needs re-running once more games have
    graded with the data actually present -- but re-running it meant someone
    remembering to do that by hand and reading a CLI table. Writing it to disk
    on the same cadence as everything else in docs/data means the dashboard
    can show it instead, so "queued" candidates are visibly rechecked every
    build rather than the next time someone happens to think of it.
    """
    if samples is None:
        samples, _centre = samples_from_log(data_dir)
    frozen = sum(1 for sample in samples if sample.frozen)
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nSamples": len(samples),
        # How many rows were fitted on features pinned before the game started.
        # Everything logged before 13 Aug 2026 was recomputed afterwards and may
        # carry the result, so until this reaches a usable size the fit is
        # trained on contaminated history and its weights are too large for the
        # signal those features actually carry pre-game. This number is the
        # countdown to a re-baseline; it cannot be backfilled.
        "frozenSamples": frozen,
        "shippedSize": len(ANCHORED_FEATURES),
        "rows": ablate(samples),
        # The nested sweep above cannot tell "useless" from "useless behind
        # seven others". These two can. See ablate_each.
        "perFeature": ablate_each(samples),
    }
    write_json(data_dir / ABLATION_FILE, payload)
    return payload


def fit_and_write(data_dir: Path, *, l2: float | None = None) -> dict[str, Any]:
    """Fit from the graded log and persist weights next to the other data files.

    An explicit ``--l2`` stays a plain shared scalar, for anyone comparing
    against the old behaviour. Left to choose for itself, the fit now picks
    strengthDiff and marketLogit their own ridge strengths rather than one
    shared penalty -- see choose_anchored_penalties for why a shared scalar
    could not express the asymmetry the data already shows.
    """
    samples, centre = samples_from_log(data_dir)
    if not samples:
        raise SystemExit("No graded samples found; cannot fit.")

    chosen_l2 = choose_anchored_penalties(samples) if l2 is None else l2
    payload = fit_from_observations(samples, l2=chosen_l2, split_diff_centre=centre)
    payload["walkForward"] = walk_forward_scores(samples, l2=chosen_l2)
    write_json(data_dir / WEIGHTS_FILE, payload)
    return payload


def _format_l2(l2: float | dict[str, float]) -> str:
    if isinstance(l2, (int, float)):
        return str(l2)
    parts = ", ".join(f"{name}={value}" for name, value in l2.items() if name != "_default")
    return f"{{{parts}}}"


def _describe(payload: dict[str, Any]) -> None:
    print(f"Fitted on {payload['metadata']['nTotal']} graded games"
          f" ({payload['metadata']['nWithMarket']} with odds),"
          f" l2={_format_l2(payload['metadata']['l2'])}")
    print(f"Split-diff centre (home-field baked into splits): {payload['splitDiffCentre']:+.4f}\n")

    for name in ("anchored", "standalone"):
        block = payload.get(name) or {}
        if not block.get("weights"):
            print(f"  {name}: not fitted (insufficient data)")
            continue
        print(f"  {name} (n={block['n']})")
        weights = block["weights"]
        print(f"    {'intercept':<14} {weights[0]:+.4f}   <- home-field advantage")
        for feature, weight in zip(block["features"], weights[1:]):
            print(f"    {feature:<14} {weight:+.4f}")
        print()

    print("  per-league intercept corrections (shrunk toward pooled)")
    for league, value in sorted((payload.get("leagueIntercepts") or {}).items()):
        print(f"    {league:<10} {value:+.4f}")

    walk = payload.get("walkForward") or {}
    if walk.get("n"):
        print(f"\n  walk-forward (n={walk['n']}, {walk['folds']} folds):"
              f" logloss {walk['logLoss']} · brier {walk['brier']} · acc {walk['accuracy']}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit win-probability coefficients from graded games.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "docs" / "data",
    )
    parser.add_argument("--l2", type=float, default=None, help="Ridge strength (default: choose by walk-forward)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and report without writing weights")
    parser.add_argument("--ablate", action="store_true", help="Walk-forward score each nested feature set")
    parser.add_argument(
        "--with-history",
        action="store_true",
        help="Include backfilled past seasons in the ablation (screening only -- no market feature)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="With --ablate, persist the table to docs/data/ablation.json (ignored with --with-history, "
        "whose mixed sample would misrepresent the standard recheck)",
    )
    args = parser.parse_args()

    if args.ablate:
        samples, centre = samples_from_log(args.data_dir)
        if args.with_history:
            history = load_history_samples(args.data_dir)
            if history:
                extra = [
                    Sample(
                        values=build_feature_dict(features, split_diff_centre=centre),
                        label=label,
                        league=league,
                        date=row_date,
                    )
                    for features, label, league, row_date in history
                ]
                samples = sorted(list(samples) + extra, key=lambda item: item.date)
                print(
                    f"Screening set: {len(extra)} backfilled games added. "
                    "These carry no market feature, so read the standalone rows "
                    "and ignore anything involving marketLogit."
                )
            else:
                print(
                    "No backfilled history found. Build some with:\n"
                    "  python scripts/backfill_history.py --league mlb "
                    "--start 2025-04-01 --end 2025-09-28"
                )
        print(f"Walk-forward ablation on {len(samples)} graded games")
        print(f"  {'features':<52} {'l2':>5} {'logloss':>8} {'brier':>8} {'acc':>7}")
        for row in ablate(samples):
            names = "+".join(row["features"])
            print(f"  {names:<52} {row['l2']:>5} {row['logLoss']:>8.4f}"
                  f" {row['brier']:>8.4f} {row['accuracy']:>7.4f}")
        if args.write:
            if args.with_history:
                print("\n--write ignored: --with-history mixes in screening-only "
                      "games with no market feature, which would misrepresent the "
                      "standard recheck on the dashboard.")
            else:
                ablate_and_write(args.data_dir, samples=samples)
                print(f"\nWrote {args.data_dir / ABLATION_FILE}")
        return 0

    if args.dry_run:
        samples, centre = samples_from_log(args.data_dir)
        chosen = choose_anchored_penalties(samples) if args.l2 is None else args.l2
        payload = fit_from_observations(samples, l2=chosen, split_diff_centre=centre)
        payload["walkForward"] = walk_forward_scores(samples, l2=chosen)
    else:
        payload = fit_and_write(args.data_dir, l2=args.l2)

    _describe(payload)
    if not args.dry_run:
        print(f"\nWrote {args.data_dir / WEIGHTS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
