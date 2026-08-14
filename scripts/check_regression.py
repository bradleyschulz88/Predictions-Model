#!/usr/bin/env python3
"""Fail the build if the model gets measurably worse.

The model refits on every scheduled run as new games are graded. Without a gate,
a bad data day or a botched change degrades the published probabilities silently
and the only symptom is a slow drift in the accuracy page. This compares the
current walk-forward score against a committed baseline and refuses to let it
slide.

Thresholds are loose enough to absorb ordinary sampling noise on a few hundred
games and tight enough to catch a real regression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BASELINE_FILE = ROOT / "docs" / "data" / "model_baseline.json"
EVALUATION_FILE = ROOT / "docs" / "data" / "evaluation.json"
WEIGHTS_FILE = ROOT / "docs" / "data" / "model_weights.json"

# The fitted intercept is meant to hold only the residual the market anchor
# misses, not home-field itself -- which is what lets NBA and NFL inherit it
# before they have a graded game of their own. Measured 2026-08-05, no league's
# entire home edge reaches this: the raw home logits run +0.1137 (mlb) to
# +0.2231 (wnba), and the fitted global intercept sits at +0.0367. An intercept
# past this band means the market feature has stopped carrying home advantage
# and the intercept has started, at which point a cold-start league really is
# being handed the wrong number. See scripts/measure_margin_sd.py.
#
# This lives here rather than in the test suite because model_weights.json is
# gitignored and written by the refit step -- a checkout has no weights to
# check, so a unit test could only ever assert against a file that never ships.
MAX_RESIDUAL_INTERCEPT = 0.25

# A league further than this from the market, in percentage points of median
# absolute gap, fails the build. Measured 13 Aug 2026: WNBA 4.5, AFL 5.6, MLB
# 8.2 -- and NFL 27.2. Set to catch the second kind without ever bothering the
# first.
MAX_LEAGUE_DIVERGENCE_PTS = 15.0
# Below this many picks a median is too easily thrown by one game to act on.
MIN_DIVERGENCE_SAMPLE = 10

# Absolute tolerances on walk-forward metrics, lower is better for both.
LOG_LOSS_TOLERANCE = 0.02
BRIER_TOLERANCE = 0.01

# A model that cannot beat a coin flip is broken regardless of the baseline.
COIN_FLIP_LOG_LOSS = 0.6931


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def check_intercepts(weights: dict[str, Any] | None = None) -> list[str]:
    """Report any fitted intercept large enough to be carrying home-field."""
    weights = _load(WEIGHTS_FILE) if weights is None else weights
    if not weights:
        return []

    failures = []
    for block in ("anchored", "standalone"):
        fitted = (weights.get(block) or {}).get("weights") or []
        if fitted and abs(fitted[0]) >= MAX_RESIDUAL_INTERCEPT:
            failures.append(
                f"{block} intercept {fitted[0]:+.4f} exceeds {MAX_RESIDUAL_INTERCEPT}"
                " -- the market feature has stopped carrying home advantage"
            )
    for league, intercept in (weights.get("leagueIntercepts") or {}).items():
        if abs(float(intercept)) >= MAX_RESIDUAL_INTERCEPT:
            failures.append(
                f"{league} intercept {float(intercept):+.4f} exceeds"
                f" {MAX_RESIDUAL_INTERCEPT} -- it is an edge, not a correction"
            )
    return failures


def check_divergence(evaluation: dict[str, Any] | None = None) -> list[str]:
    """No league may sit this far from the market without failing the build.

    Divergence from the closing line is the model's own headline claim about
    itself, and nothing tested it per league. The published figure -- median
    3.7 points with 1.9% over 15 -- is pooled and dominated by baseball, and on
    13 Aug 2026 it was concealing NFL preseason running at 27.2 points with 70%
    over 15. That is the defect the whole project was built to remove, alive in
    the one league nobody had looked at, found only because someone asked.

    The threshold is deliberately loose. This is not a quality bar, it is a
    tripwire for a league whose inputs have gone wrong -- a cold start, a
    provider returning nonsense, an exhibition read as a real fixture. Every
    working league sits between 4.5 and 8.2, so 15 leaves plenty of room for a
    genuine disagreement and still catches the case that matters.

    Small samples are exempt: a league with a handful of picks can post a wild
    median honestly, and failing the build over three games would train
    everyone to ignore this.
    """
    report = evaluation if evaluation is not None else _load(EVALUATION_FILE)
    per_league = (report.get("divergence") or {}).get("byLeague") or {}
    failures: list[str] = []
    for league, stats in sorted(per_league.items()):
        median = stats.get("medianGapPct")
        count = stats.get("n") or 0
        if median is None or count < MIN_DIVERGENCE_SAMPLE:
            continue
        if median > MAX_LEAGUE_DIVERGENCE_PTS:
            failures.append(
                f"{league} sits a median {median:.1f}pts from the market on {count} picks "
                f"(limit {MAX_LEAGUE_DIVERGENCE_PTS}); its inputs are probably wrong, not its opinion"
            )
    return failures


def check(*, update: bool = False) -> int:
    intercept_failures = check_intercepts()
    if intercept_failures:
        print("FAIL: a fitted intercept has grown past a residual")
        for failure in intercept_failures:
            print(f"  - {failure}")
        return 1

    divergence_failures = check_divergence()
    if divergence_failures:
        print("FAIL: a league has drifted away from the market")
        for failure in divergence_failures:
            print(f"  - {failure}")
        return 1

    evaluation = _load(EVALUATION_FILE)
    current = evaluation.get("fittedWalkForward") or {}
    if not current.get("logLoss"):
        print("No walk-forward score available; skipping regression check.")
        return 0

    log_loss = float(current["logLoss"])
    brier = float(current["brier"])
    print(f"Walk-forward: logloss {log_loss:.4f} · brier {brier:.4f} · n={current.get('n')}")

    if log_loss >= COIN_FLIP_LOG_LOSS:
        print(f"FAIL: log loss {log_loss:.4f} is no better than a coin flip ({COIN_FLIP_LOG_LOSS}).")
        return 1

    baseline = _load(BASELINE_FILE)
    if update or not baseline:
        BASELINE_FILE.write_text(
            json.dumps({"logLoss": log_loss, "brier": brier, "n": current.get("n")}, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote baseline to {BASELINE_FILE}")
        return 0

    failures: list[str] = []
    if log_loss > baseline["logLoss"] + LOG_LOSS_TOLERANCE:
        failures.append(
            f"log loss {log_loss:.4f} exceeds baseline {baseline['logLoss']:.4f}"
            f" by more than {LOG_LOSS_TOLERANCE}"
        )
    if brier > baseline["brier"] + BRIER_TOLERANCE:
        failures.append(
            f"brier {brier:.4f} exceeds baseline {baseline['brier']:.4f}"
            f" by more than {BRIER_TOLERANCE}"
        )

    if failures:
        print("FAIL: model regressed against committed baseline")
        for failure in failures:
            print(f"  - {failure}")
        print("\nIf the change is intended, re-run with --update-baseline and commit the result.")
        return 1

    print(f"OK: within tolerance of baseline (logloss {baseline['logLoss']:.4f}, brier {baseline['brier']:.4f})")

    # Ratchet the baseline down when the model genuinely improves, so a later
    # regression is measured against the best score achieved rather than the
    # first one recorded.
    if log_loss < baseline["logLoss"]:
        BASELINE_FILE.write_text(
            json.dumps({"logLoss": log_loss, "brier": brier, "n": current.get("n")}, indent=2),
            encoding="utf-8",
        )
        print(f"Improved on baseline; ratcheted to logloss {log_loss:.4f}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate the build on model quality.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the committed baseline with the current score",
    )
    args = parser.parse_args()
    return check(update=args.update_baseline)


if __name__ == "__main__":
    raise SystemExit(main())
