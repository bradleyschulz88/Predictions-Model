#!/usr/bin/env python3
"""
Evaluate the robustly trained XGBoost model with detailed analysis.
Uses the EXACT same feature engineering as train_robust.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = ROOT / "docs" / "data"

MODEL_FILE = MODEL_DIR / "xgboost_model.pkl"
CALIBRATOR_FILE = MODEL_DIR / "calibrator.pkl"
PREDICTIONS_LOG = ROOT / "docs" / "data" / "predictions_log.json"
ACCURACY_FILE = ROOT / "docs" / "data" / "accuracy.json"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """EXACT copy from train_robust.py"""
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


def prepare_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Prepare features using the EXACT training columns."""
    df = engineer_features(df)
    
    # Ensure all training columns exist (fill missing with 0)
    X = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0  # Will be filled with median later
    
    # Only numeric
    X = X.select_dtypes(include=[np.number])
    # Fill missing with median
    X = X.fillna(X.median())
    # Ensure column order
    X = X[feature_columns]
    return X


def load_model():
    with open("ml_model/models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ml_model/models/calibrator.pkl", "rb") as f:
        cal_data = pickle.load(f)
    return model, cal_data["calibrator"]


def load_data():
    with open("docs/data/predictions_log.json", "r") as f:
        log_data = json.load(f)
    with open("docs/data/accuracy.json", "r") as f:
        acc_data = json.load(f)
    
    predictions = log_data.get("predictions", {})
    picks_by_event = acc_data.get("picksByEventId", {})
    
    rows = []
    for event_id, pred in predictions.items():
        features = pred.get("features", {})
        if not features:
            continue
        acc_record = picks_by_event.get(event_id, {})
        if acc_record.get("status") != "graded":
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
            "confidence": pred.get("confidence", 0),
            "correct": int(correct),
        }
        
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
                    row[f"{k}_mean"] = np.mean(v) if v else 0
                    row[f"{k}_len"] = len(v)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df["schedule_date"] = pd.to_datetime(df["schedule_date"], errors="coerce")
    df = df.sort_values("schedule_date").reset_index(drop=True)
    return df


def main():
    print("=" * 60)
    print("Comprehensive Model Evaluation (matched features)")
    print("=" * 60)
    
    model, calibrator = load_model()
    
    # Load feature columns from metadata
    with open("ml_model/models/model_metadata.json", "r") as f:
        metadata = json.load(f)
    feature_columns = metadata["feature_columns"]
    
    print(f"Training feature columns: {len(feature_columns)}")
    
    df = load_data()
    df = df.sort_values("schedule_date").reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    y_test = test_df["correct"].astype(int)
    
    print(f"Test set: {len(test_df)} games")
    print(f"Win rate: {y_test.mean():.3f}")
    
    # Prepare features using EXACT training columns
    test_df_eng = engineer_features(test_df.copy())
    X_test = pd.DataFrame(index=test_df.index)
    for col in feature_columns:
        if col in test_df_eng.columns:
            X_test[col] = test_df_eng[col]
        else:
            X_test[col] = 0
    X_test = X_test.select_dtypes(include=[np.number]).fillna(0)[feature_columns]
    
    y_test = test_df["correct"].astype(int)
    
    # Predictions
    xgb_proba = calibrator.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    
    # Metrics
    xgb_metrics = {
        "log_loss": log_loss(y_test, xgb_proba),
        "brier": brier_score_loss(y_test, xgb_proba),
        "auc": roc_auc_score(y_test, xgb_proba),
        "accuracy": accuracy_score(y_test, xgb_pred),
    }
    prior = y_test.mean()
    baseline_metrics = {
        "log_loss": log_loss(y_test, np.full_like(y_test, prior)),
        "brier": brier_score_loss(y_test, np.full_like(y_test, prior)),
        "auc": 0.5,
        "accuracy": max(prior, 1 - prior),
    }
    
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    for name, m in [("Baseline (prior)", baseline_metrics), ("XGBoost (calibrated)", xgb_metrics)]:
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")
    
    print("\nIMPROVEMENT OVER BASELINE:")
    for k in xgb_metrics:
        if k != "auc":
            imp = (baseline_metrics[k] - xgb_metrics[k]) / baseline_metrics[k] * 100
            print(f"  {k}: {imp:+.1f}%")
        else:
            print(f"  {k}: {xgb_metrics[k]:.4f} (baseline: 0.5000)")
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, xgb_proba, n_bins=15, strategy="quantile")
    print("\nQuantile Bins:")
    for i in range(len(prob_true)):
        mask = (xgb_proba >= np.percentile(xgb_proba, i/15*100)) & (xgb_proba < np.percentile(xgb_proba, (i+1)/15*100))
        if i == 14:
            mask = xgb_proba >= np.percentile(xgb_proba, 100*14/15)
        count = mask.sum()
        if count > 0:
            print(f"  [{prob_pred[i]:.3f}]: pred={prob_pred[i]:.3f}, actual={prob_true[i]:.3f}, n={count}")
    
    # By confidence bins
    print("\nBy Confidence Bins:")
    for conf_range in [(0,55), (55,60), (60,65), (65,70), (70,80), (80,100)]:
        mask = (test_df["confidence"] >= conf_range[0]) & (test_df["confidence"] < conf_range[1])
        if mask.sum() > 0:
            actual = y_test[mask].mean()
            pred = xgb_proba[mask].mean()
            print(f"  [{conf_range[0]}-{conf_range[1]}): pred={pred:.3f}, actual={actual:.3f}, n={mask.sum()}")
    
    # By league
    print("\nBy League:")
    for league in test_df["league"].unique():
        mask = test_df["league"] == league
        if mask.sum() > 10:
            actual = y_test[mask].mean()
            pred = xgb_proba[mask].mean()
            acc = accuracy_score(y_test[mask], (xgb_proba[mask]>=0.5).astype(int))
            print(f"  {league}: pred={pred:.3f}, actual={actual:.3f}, n={mask.sum()}, acc={acc:.3f}")
    
    # Save detailed predictions
    results_df = pd.DataFrame({
        "event_id": test_df["event_id"],
        "date": test_df["schedule_date"],
        "league": test_df["league"],
        "home_team": test_df["home_team"],
        "away_team": test_df["away_team"],
        "home_win": y_test,
        "xgb_proba": xgb_proba,
        "xgb_pred": xgb_pred,
        "confidence": test_df["confidence"],
    })
    results_df.to_csv("ml_model/data/test_predictions_detailed.csv", index=False)
    print("\nDetailed predictions saved to ml_model/data/test_predictions_detailed.csv")


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import pickle
    import json
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score
    from sklearn.calibration import calibration_curve
    import warnings
    warnings.filterwarnings("ignore")
    
    ROOT = Path(__file__).parent.parent.parent
    MODEL_DIR = Path(__file__).parent.parent / "models"
    DATA_DIR = ROOT / "docs" / "data"
    
    MODEL_FILE = Path("ml_model/models/xgboost_model.pkl")
    CALIBRATOR_FILE = Path("ml_model/models/calibrator.pkl")
    PREDICTIONS_LOG = Path("docs/data/predictions_log.json")
    ACCURACY_FILE = Path("docs/data/accuracy.json")
    
    main()