#!/usr/bin/env python3
"""
XGBoost model training with temporal cross-validation.
Replaces hand-tuned logit adjustments with learned features.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_FILE = DATA_DIR / "training_features.parquet"
MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
CALIBRATOR_FILE = MODEL_DIR / "calibrator.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


# Feature columns to use (exclude identifiers, targets, leaky features)
EXCLUDE_COLS = {
    "event_id", "date", "league", "season", "home_team", "away_team",
    "home_win", "home_score", "away_score",  # targets
    "home_form_results", "away_form_results",  # text
    "market_edge_home",  # leakage - uses home_win_pct which is computed from same game
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and prepare training data."""
    df = pd.read_parquet(FEATURE_FILE)
    
    # Filter to MLB for now (most data)
    df = df[df["league"] == "mlb"].copy()
    
    # Sort by date for temporal splits
    df = df.sort_values("date").reset_index(drop=True)
    
    # Target
    y = df["home_win"].astype(int)
    
    # Features
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median(numeric_only=True))
    
    print(f"Loaded {len(df)} games, {len(feature_cols)} features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Home win rate: {y.mean():.3f}")
    
    return X, y


def temporal_cv_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create temporal cross-validation splits (no shuffling)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))


def train_fold(X_train: pd.DataFrame, y_train: pd.Series, 
               X_val: pd.DataFrame, y_val: pd.Series,
               params: dict) -> tuple[xgb.XGBClassifier, dict]:
    """Train a single fold."""
    model = xgb.XGBClassifier(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    
    # Predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        "log_loss": log_loss(y_val, y_pred_proba),
        "brier": brier_score_loss(y_val, y_pred_proba),
        "auc": roc_auc_score(y_val, y_pred_proba),
        "accuracy": accuracy_score(y_val, y_pred),
    }
    
    return model, metrics


def main():
    print("=" * 60)
    print("XGBoost Model Training with Temporal CV")
    print("=" * 60)
    
    # Load data
    X, y = load_data()
    
    # XGBoost parameters (tuned for sports prediction)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "enable_categorical": False,
    }
    
    # Temporal cross-validation
    print("\nRunning temporal cross-validation...")
    splits = temporal_cv_split(X, y, n_splits=5)
    
    fold_metrics = []
    models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\nFold {fold_idx + 1}/{len(splits)}:")
        print(f"  Train: {len(train_idx)} games ({X.iloc[train_idx]['date'].min()} to {X.iloc[train_idx]['date'].max()})")
        print(f"  Val:   {len(val_idx)} games ({X.iloc[val_idx]['date'].min()} to {X.iloc[val_idx]['date'].max()})")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model, metrics = train_fold(X_train, y_train, X_val, y_val, params)
        
        print(f"  LogLoss: {metrics['log_loss']:.4f} | Brier: {metrics['brier']:.4f} | AUC: {metrics['auc']:.4f} | Acc: {metrics['accuracy']:.4f}")
        
        fold_metrics.append(metrics)
        models.append(model)
    
    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for k in avg_metrics:
        print(f"  {k}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
    
    # Train final model on ALL data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    
    # Calibrate with isotonic regression
    print("Calibrating probabilities...")
    calibrator = CalibratedClassifierCV(final_model, method="isotonic", cv=3)
    calibrator.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 20 Features:")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print(f"\nSaving model to {MODEL_FILE}")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    
    print(f"Saving calibrator to {CALIBRATOR_FILE}")
    with open(CALIBRATOR_FILE, "wb") as f:
        pickle.dump(calibrator, f)
    
    # Save metadata
    metadata = {
        "model_type": "XGBoost",
        "params": params,
        "cv_metrics": {k: {"mean": float(avg_metrics[k]), "std": float(std_metrics[k])} for k in avg_metrics},
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
        "n_training_samples": len(X),
        "date_range": {"min": str(X["date"].min()), "max": str(X["date"].max())},
        "home_win_rate": float(y.mean()),
        "feature_importance": importance.to_dict("records"),
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {METADATA_FILE}")
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()