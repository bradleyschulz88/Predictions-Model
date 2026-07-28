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


def check(*, update: bool = False) -> int:
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
