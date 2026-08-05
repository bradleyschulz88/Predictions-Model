# Experimental training scripts — nothing here runs automatically

None of these are on the prediction path, none are imported by the live model,
and none run in CI. They are the reproduction path for the artifacts in
`ml_model/experimental/`, kept so those `.pkl` files are not unreproducible
binaries. See `ml_model/experimental/README.md` for why those are quarantined
rather than deleted, and for the `model_fit.ProbabilityModel` seam to use if a
model like this ever earns its way back onto the prediction path.

## They will not run as the repo stands

They import `numpy`, `scikit-learn` and `xgboost`. The runtime is stdlib-only by
design and those are not in `requirements.txt`, so running any of these needs a
separate environment:

    python -m venv .venv-ml && .venv-ml/bin/pip install numpy scikit-learn xgboost

## What each one does

| script | role |
|---|---|
| `extract_features.py` | builds a training matrix from the graded log |
| `generate_synthetic_training.py` | pads that matrix with synthetic rows |
| `train_model.py` | fits the baseline gradient-boosted model |
| `train_robust.py` | fits the variant with heavier regularisation |
| `train_from_logs.py` | retrains from `docs/data/predictions_log.json` |
| `evaluate_model.py` | out-of-fold metrics for the baseline fit |
| `evaluate_robust.py` | the same for the robust fit |

`train_from_logs.py` calls into `train_model.py`, and `evaluate_robust.py` into
`train_robust.py` — so the pairs live and die together.

## Before trusting anything they print

Their own metrics are not comparable to the live model's: different sample,
synthetic rows included, per-fold AUC ranging 0.47–0.66 at n=510. The only
comparison that counts is `python model_fit.py --ablate` and
`python scripts/check_regression.py` against the walk-forward baseline in
`docs/data/model_baseline.json`.
