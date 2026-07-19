#!/usr/bin/env python3
"""
Train XGBoost model from historical predictions_log.json and accuracy.json.
Uses actual historical predictions and their graded outcomes.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "docs" / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

PREDICTIONS_LOG = DATA_DIR / "predictions_log.json"
ACCURACY_FILE = DATA_DIR / "accuracy.json"
MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
CALIBRATOR_FILE = MODEL_DIR / "calibrator.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


def load_data() -> pd.DataFrame:
    """Load and merge predictions_log.json with accuracy.json."""
    
    with open(PREDICTIONS_LOG, "r") as f:
        log_data = json.load(f)
    
    with open(ACCURACY_FILE, "r") as f:
        acc_data = json.load(f)
    
    predictions = log_data.get("predictions", {})
    picks_by_event = acc_data.get("summary", {}).get("picksByEventId", {})
    # The accuracy.json structure has picksByEventId at the top level
    if "picksByEventId" in acc_data:
        picks_by_event = acc_data["picksByEventId"]
    
    rows = []
    
    for event_id, pred in predictions.items():
        # Get features
        features = pred.get("features", {})
        if not features:
            continue
        
        # Get outcome from accuracy
        acc_record = picks_by_event.get(event_id, {})
        status = acc_record.get("status", "pending")
        
        if status != "graded":
            continue  # Only use graded predictions
        
        correct = acc_record.get("correct")
        if correct is None:
            continue
        
        # Extract features into flat dict
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
    
    # Convert date
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.sort_values("schedule_date").reset_index(drop=True)
    
    print(f"Loaded {len(df)} graded predictions")
    print(f"Leagues: {df['league'].value_counts().to_dict()}")
    print(f"Correct rate: {df['correct'].mean():.3f}")
    print(f"Date range: {df['schedule_date'].min()} to {df['schedule_date'].max()}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target."""
    
    # Target
    y = df["correct"].astype(int)
    
    # Feature columns (exclude identifiers, targets, text)
    exclude_cols = {
        "event_id", "league", "schedule_date", "home_team", "away_team",
        "predicted_side", "pick_odds", "correct",
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    
    # Handle missing values
    # Separate numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    print(f"\nFeatures: {len(X.columns)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[xgb.XGBClassifier, CalibratedClassifierCV, dict]:
    """Train XGBoost with temporal CV."""
    
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
    }
    
    # Temporal cross-validation
    print("\nRunning temporal cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            "log_loss": log_loss(y_val, y_pred_proba),
            "brier": brier_score_loss(y_val, y_pred_proba),
            "auc": roc_auc_score(y_val, y_pred_proba),
            "accuracy": accuracy_score(y_val, y_pred),
        }
        
        fold_metrics.append(metrics)
        print(f"  Fold {fold_idx+1}: LogLoss={metrics['log_loss']:.4f}, Brier={metrics['brier']:.4f}, AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    # Average metrics
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    
    print("\n" + "=" * 50)
    print("CV RESULTS:")
    for k in avg_metrics:
        print(f"  {k}: {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    
    # Calibrate
    print("Calibrating...")
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
    
    return final_model, calibrator, {
        "avg_metrics": avg_metrics,
        "std_metrics": std_metrics,
        "importance": importance.to_dict("records"),
    }


def save_model(model: xgb.XGBClassifier, calibrator: CalibratedClassifierCV, metrics: dict, X: pd.DataFrame, y: pd.Series):
    """Save model and metadata."""
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    with open(CALIBRATOR_FILE, "wb") as f:
        pickle.dump(calibrator, f)
    
    metadata = {
        "model_type": "XGBoost",
        "cv_metrics": {k: {"mean": float(metrics["avg_metrics"][k]), "std": float(metrics["std_metrics"][k])} for k in metrics["avg_metrics"]},
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
        "n_training_samples": len(X),
        "home_win_rate": float(y.mean()),
        "feature_importance": metrics["importance"],
        "trained_at": datetime.now().isoformat(),
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Calibrator saved to {CALIBRATOR_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")


def main():
    print("=" * 60)
    print("Training XGBoost from Historical Predictions")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    if len(df) < 50:
        print("Not enough graded predictions for training!")
        return
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Train
    model, calibrator, metrics = train_model(X, y)
    
    # Save
    save_model(model, calibrator, metrics, X, y)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()