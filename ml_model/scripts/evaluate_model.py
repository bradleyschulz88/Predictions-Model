#!/usr/bin/env python3
"""
Evaluate trained XGBoost model against baselines.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent.parent / "docs" / "data"

MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
CALIBRATOR_FILE = MODEL_DIR / "calibrator.pkl"
FEATURE_FILE = DATA_DIR / "predictions_log.json"
ACCURACY_FILE = DATA_DIR / "accuracy.json"


EXCLUDE_COLS = {
    "event_id", "league", "schedule_date", "home_team", "away_team",
    "predicted_side", "pick_odds", "correct",
}


def load_model() -> tuple[xgb.XGBClassifier, CalibratedClassifierCV]:
    """Load trained XGBoost model and calibrator."""
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(CALIBRATOR_FILE, "rb") as f:
        calibrator = pickle.load(f)
    return model, calibrator


def load_data() -> pd.DataFrame:
    """Load and merge predictions_log.json with accuracy.json."""
    
    with open(FEATURE_FILE, "r") as f:
        log_data = json.load(f)
    
    with open(ACCURACY_FILE, "r") as f:
        acc_data = json.load(f)
    
    predictions = log_data.get("predictions", {})
    picks_by_event = acc_data.get("picksByEventId", {})
    
    rows = []
    
    for event_id, pred in predictions.items():
        features = pred.get("features", {})
        if not features:
            continue
        
        acc_record = picks_by_event.get(event_id, {})
        status = acc_record.get("status", "pending")
        
        if status != "graded":
            continue
        
        correct = acc_record.get("correct")
        if correct is None:
            continue
        
        row = {
            "event_id": event_id,
            "league": pred.get("league", "unknown"),
            "schedule_date": pred.get("scheduleDate", ""),
            "home_team": pred.get("homeTeam", ""),
            "away_team": pred.get("awayTeam", ""),
            "predicted_side": pred.get("predictedSide", ""),
            "confidence": pred.get("confidence", 0),
            "pick_odds": pred.get("pickOdds"),
            "correct": int(correct),
        }
        
        # Flatten features
        for k, v in features.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float, bool)) and sv is not None:
                        row[f"{k}_{sk}"] = sv
            elif isinstance(v, (int, float, bool)) and v is not None:
                row[k] = v
            elif isinstance(v, list):
                if all(isinstance(x, bool) for x in v):
                    row[f"{k}_count"] = sum(v)
                elif all(isinstance(x, (int, float)) for x in v):
                    row[f"{k}_sum"] = sum(v)
                    row[f"{k}_mean"] = np.mean(v) if v else 0
                    row[f"{k}_len"] = len(v)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.sort_values("schedule_date").reset_index(drop=True)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix."""
    exclude_cols = {
        "event_id", "league", "schedule_date", "home_team", "away_team",
        "predicted_side", "pick_odds", "correct",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X = X.select_dtypes(include=[np.number])
    return X


def main():
    print("=" * 60)
    print("Model Evaluation: XGBoost vs Baselines")
    print("=" * 60)
    
    # Load data
    df = load_data()
    df = df.sort_values("schedule_date").reset_index(drop=True)
    
    # Use last 20% as holdout test set
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    y_test = test_df["correct"].astype(int)
    
    print(f"Test set: {len(test_df)} games")
    print(f"Date range: {test_df['schedule_date'].min()} to {test_df['schedule_date'].max()}")
    print(f"Home win rate: {y_test.mean():.3f}")
    
    # Load XGBoost model
    model, calibrator = load_model()
    
    # Prepare features
    X_test = prepare_features(test_df)
    
    # XGBoost predictions (calibrated)
    xgb_proba = calibrator.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    
    # Baseline: always predict home win rate (or class prior)
    prior = y_test.mean()
    baseline_proba = np.full_like(y_test, prior, dtype=float)
    
    # XGBoost metrics
    xgb_metrics = {
        "log_loss": log_loss(y_test, xgb_proba),
        "brier": brier_score_loss(y_test, xgb_proba),
        "auc": roc_auc_score(y_test, xgb_proba),
        "accuracy": accuracy_score(y_test, xgb_pred),
    }
    
    # Baseline metrics
    baseline_metrics = {
        "log_loss": log_loss(y_test, baseline_proba),
        "brier": brier_score_loss(y_test, baseline_proba),
        "auc": 0.5,
        "accuracy": max(prior, 1 - prior),
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    
    for name, metrics in [("Baseline (class prior)", baseline_metrics), 
                           ("XGBoost (calibrated)", xgb_metrics)]:
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Improvement
    print("\n" + "-" * 60)
    print("IMPROVEMENT OVER BASELINE:")
    for k in xgb_metrics:
        if k != "auc":
            imp = (baseline_metrics[k] - xgb_metrics[k]) / baseline_metrics[k] * 100
            print(f"  {k}: {imp:+.1f}%")
        else:
            print(f"  {k}: {xgb_metrics[k]:.4f} (baseline: 0.5000)")
    
    # Calibration check
    print("\n" + "-" * 60)
    print("CALIBRATION CHECK (XGBoost):")
    bins = np.linspace(0, 1, 11)
    for i in range(len(bins) - 1):
        mask = (xgb_proba >= bins[i]) & (xgb_proba < bins[i + 1])
        if mask.sum() > 0:
            avg_pred = xgb_proba[mask].mean()
            actual = y_test[mask].mean()
            count = mask.sum()
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): pred={avg_pred:.3f}, actual={actual:.3f}, n={count}")
    
    # Save predictions
    results_df = pd.DataFrame({
        "event_id": test_df["event_id"],
        "date": test_df["schedule_date"],
        "home_team": test_df["home_team"],
        "away_team": test_df["away_team"],
        "home_win": y_test,
        "xgb_proba": xgb_proba,
        "xgb_pred": xgb_pred,
    })
    
    output_file = Path(__file__).parent.parent / "data" / "test_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    main()