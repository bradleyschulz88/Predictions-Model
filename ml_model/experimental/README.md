# Experimental artifacts — not on the prediction path

These files are **not used** by the live model. Nothing in the prediction path
imports them; only the training and evaluation scripts under `ml_model/scripts/`
reference them, and those are run manually.

## What they are

A gradient-boosted model and isotonic calibrator trained on 510 samples, partly
synthetic (see `ml_model/scripts/generate_synthetic_training.py`). Reported
out-of-fold metrics from `model_metadata.json`:

| metric | value |
|---|---|
| log loss | 0.582 |
| Brier | 0.206 |
| AUC | 0.738 |
| accuracy | 0.661 |

Those numbers look better than the live model's, but they are not comparable:
they come from a different sample, include synthetic rows, and the per-fold AUCs
range from 0.47 to 0.66 — wide enough that the aggregate is not trustworthy at
n=510.

## Why they are quarantined rather than deleted

The runtime is stdlib-only by design; loading these requires numpy, scikit-learn
and xgboost, which would add roughly 200MB to every CI run. They are kept
because the approach is worth revisiting once the graded log is larger.

## How to put a model like this back on the prediction path

`model_fit.ProbabilityModel` is the seam. Implement
`predict_proba(features, league) -> float | None`, have `model_core.get_fitted_model`
return it, and add the dependencies to `requirements.txt` and the workflow.

Before doing that, make it earn its place the same way every current feature had
to — `python model_fit.py --ablate` and `python scripts/check_regression.py`
compare against the live walk-forward baseline in `docs/data/model_baseline.json`.
A model that cannot beat that baseline out of sample should not ship regardless
of what its own training metrics say.
