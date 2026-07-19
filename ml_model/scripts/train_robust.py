#!/usr/bin/env python3
"""
Enhanced XGBoost training with proper calibration, feature engineering, and validation.
Addresses calibration issues and integrates with the actual prediction pipeline.
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


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
PLOTS_DIR = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load and merge predictions_log.json with accuracy.json."""

    with open(PREDICTIONS_LOG, "r") as f:
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

        # Flatten features with better handling
        for k, v in features.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float, bool)) and sv is not None:
                        row[f"{k}_{sk}"] = float(sv) if isinstance(sv, (int, float)) else sv
            elif isinstance(v, (int, float, bool)) and v is not None:
                row[k] = float(v) if isinstance(v, (int, float)) else v
            elif isinstance(v, list):
                if all(isinstance(x, bool) for x in v):
                    row[f"{k}_count"] = sum(v)
                elif all(isinstance(x, (int, float)) for x in v):
                    row[f"{k}_sum"] = sum(v)
                    row[f"{k}_mean"] = float(np.mean(v)) if v else 0.0
                    row[f"{k}_len"] = len(v)

        rows.append(row)

    df = pd.DataFrame(rows)
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.sort_values("schedule_date").reset_index(drop=True)

    print(f"Loaded {len(df)} graded predictions")
    print(f"Leagues: {df['league'].value_counts().to_dict()}")
    print(f"Correct rate: {df['correct'].mean():.3f}")
    print(f"Date range: {df['schedule_date'].min()} to {df['schedule_date'].max()}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features from raw data."""

    df = df.copy()

    # 1. Confidence-based features
    df["confidence_squared"] = df["confidence"] ** 2
    df["confidence_log"] = np.log1p(df["confidence"])
    df["confidence_bin"] = pd.cut(df["confidence"], bins=[0, 55, 60, 65, 70, 80, 100], labels=False)

    # 2. Implied odds features
    if "impliedHome" in df.columns and "impliedAway" in df.columns:
        df["implied_total"] = df["impliedHome"] + df["impliedAway"]
        df["implied_diff"] = df["impliedHome"] - df["impliedAway"]
        df["implied_entropy"] = -(
            (df["impliedHome"]/100) * np.log(np.clip(df["impliedHome"]/100, 1e-6, 1)) +
            (df["impliedAway"]/100) * np.log(np.clip(df["impliedAway"]/100, 1e-6, 1))
        )

    # 3. True probability features
    if "trueHome" in df.columns and "trueAway" in df.columns:
        df["true_total"] = df["trueHome"] + df["trueAway"]
        df["true_diff"] = df["trueHome"] - df["trueAway"]
        df["model_edge_home"] = df["trueHome"] - df.get("impliedHome", 0)
        df["model_edge_away"] = df["trueAway"] - df.get("impliedAway", 0)
        df["model_edge_max"] = np.maximum(df["model_edge_home"], df["model_edge_away"])

    # 4. Power rating features
    if "homePower" in df.columns and "awayPower" in df.columns:
        df["power_diff"] = df["homePower"] - df["awayPower"]
        df["power_ratio"] = df["homePower"] / df["awayPower"].replace(0, np.nan)
        df["power_sum"] = df["homePower"] + df["awayPower"]

    # 5. Pitching features (MLB)
    pitch_cols = [c for c in df.columns if "Pitcher" in c or "Pitching" in c or "Bullpen" in c or "Era" in c]
    if len(pitch_cols) > 1:
        home_pitch = [c for c in pitch_cols if "home" in c.lower() or "Home" in c]
        away_pitch = [c for c in pitch_cols if "away" in c.lower() or "Away" in c]
        if home_pitch and away_pitch:
            df["pitching_era_diff"] = df[home_pitch[0]] - df[away_pitch[0]] if len(home_pitch) == 1 and len(away_pitch) == 1 else 0

    # 5. League metrics
    league_cols = [c for c in df.columns if c.startswith("leagueMetrics_")]
    if league_cols:
        df["league_metrics_count"] = df[league_cols].notna().sum(axis=1)
        df["league_metrics_mean"] = df[league_cols].mean(axis=1)

    # 6. Injury features
    if "homeInjuryLoad" in df.columns and "awayInjuryLoad" in df.columns:
        df["injury_load_diff"] = df["homeInjuryLoad"] - df["awayInjuryLoad"]
        df["total_injury_load"] = df["homeInjuryLoad"] + df["awayInjuryLoad"]

    # 7. Rest features
    if "homeRest" in df.columns and "awayRest" in df.columns:
        df["rest_diff"] = df["homeRest"] - df["awayRest"]
        df["min_rest"] = np.minimum(df["homeRest"], df["awayRest"])

    # 8. Back-to-back features
    if "homeBackToBack" in df.columns:
        df["home_b2b"] = df["homeBackToBack"].astype(int)
    if "awayBackToBack" in df.columns:
        df["away_b2b"] = df["awayBackToBack"].astype(int)
    if "home_b2b" in df.columns and "away_b2b" in df.columns:
        df["b2b_diff"] = df["home_b2b"] - df["away_b2b"]

    # 9. Confidence interactions
    df["conf_x_implied_diff"] = df["confidence"] * df.get("implied_diff", 0) / 100
    df["conf_x_power_diff"] = df["confidence"] * df.get("power_diff", 0) / 100
    if "model_edge_max" in df.columns:
        df["conf_x_edge"] = df["confidence"] * df["model_edge_max"] / 100

    # 10. League-specific features
    df["is_mlb"] = (df["league"] == "mlb").astype(int)
    df["is_wnba"] = (df["league"] == "wnba").astype(int)
    df["is_afl"] = (df["league"] == "afl").astype(int)
    df["is_worldcup"] = (df["league"] == "worldcup").astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target with proper handling."""

    df = engineer_features(df)

    # Target
    y = df["correct"].astype(int)

    # Exclude columns
    exclude_cols = {
        "event_id", "league", "schedule_date", "home_team", "away_team",
        "predicted_side", "pick_odds", "correct",
    }

    # Only numeric features
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    # Handle missing values
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    # Remove constant/near-constant features
    nunique = X.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print(f"  Dropped {len(constant_cols)} constant features")

    # Remove highly correlated features (>0.98)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
    if to_drop:
        X = X.drop(columns=to_drop)
        print(f"  Dropped {len(to_drop)} highly correlated features")

    print(f"  Final features: {len(X.columns)}")

    return X, y


def train_with_calibration(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train XGBoost with proper calibration using nested CV."""

    # XGBoost parameters - tuned for calibration
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 600,
        "max_depth": 4,           # Shallower for better calibration
        "learning_rate": 0.02,    # Lower LR for better calibration
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,    # Higher for regularization
        "gamma": 0.2,
        "reg_alpha": 0.5,         # L1 regularization
        "reg_lambda": 2.0,        # L2 regularization
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "max_delta_step": 1,      # Helps with class imbalance
    }

    # Temporal cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    fold_results = []
    oof_preds = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    print("\nRunning Temporal Cross-Validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_proba
        oof_true[val_idx] = y_val.values

        val_pred = (val_proba >= 0.5).astype(int)

        metrics = {
            "log_loss": log_loss(y_val, val_proba),
            "brier": brier_score_loss(y_val, val_proba),
            "auc": roc_auc_score(y_val, val_proba),
            "accuracy": accuracy_score(y_val, val_pred),
        }

        fold_results.append(metrics)
        print(f"  Fold {fold_idx+1}: LogLoss={metrics['log_loss']:.4f}, Brier={metrics['brier']:.4f}, AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")

    # OOF metrics
    oof_metrics = {
        "log_loss": log_loss(oof_true, oof_preds),
        "brier": brier_score_loss(oof_true, oof_preds),
        "auc": roc_auc_score(oof_true, oof_preds),
        "accuracy": accuracy_score(oof_true, (oof_preds >= 0.5).astype(int)),
    }

    print("\n" + "=" * 50)
    print("OOF METRICS (Uncalibrated):")
    for k, v in oof_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Calibration analysis
    print("\nCalibration Analysis (Uncalibrated):")
    prob_true, prob_pred = calibration_curve(oof_true, oof_preds, n_bins=10, strategy="quantile")
    for i in range(len(prob_true)):
        print(f"  Bin {i}: pred={prob_pred[i]:.3f}, actual={prob_true[i]:.3f}")

    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    # Calibration
    print("\nCalibrating...")
    calibrator = CalibratedClassifierCV(final_model, method="isotonic", cv=5)
    calibrator.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 20 Features:")
    for _, row in importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return {
        "model": final_model,
        "calibrator": calibrator,
        "importance": importance,
        "oof_preds": oof_preds,
        "oof_true": oof_true,
        "oof_metrics": oof_metrics,
        "fold_results": [],
    }


def save_artifacts(result: dict, X: pd.DataFrame, y: pd.Series):
    """Save model, calibrator, and metadata."""

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(result["model"], f)

    with open(CALIBRATOR_FILE, "wb") as f:
        pickle.dump({
            "calibrator": result["calibrator"],
            "type": "isotonic_cv",
            "feature_columns": list(X.columns),
        }, f)

    metadata = {
        "model_type": "XGBoost",
        "calibration_method": "isotonic_cv",
        "feature_columns": list(X.columns),
        "n_features": len(X.columns),
        "n_training_samples": len(X),
        "home_win_rate": float(y.mean()),
        "feature_importance": result["importance"].to_dict("records"),
        "oof_metrics": {k: float(v) for k, v in result["oof_metrics"].items()},
        "fold_results": [],
        "trained_at": datetime.now().isoformat(),
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Calibrator saved to {CALIBRATOR_FILE}")
    print(f"Metadata saved to {METADATA_FILE}")


def main():
    print("=" * 60)
    print("Enhanced XGBoost Training with Proper Calibration")
    print("=" * 60)

    # Load
    df = load_data()

    if len(df) < 100:
        print("Not enough data!")
        return

    # Prepare
    X, y = prepare_features(df)

    # Train
    result = train_with_calibration(X, y)

    # Save
    save_artifacts(result, X, y)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()