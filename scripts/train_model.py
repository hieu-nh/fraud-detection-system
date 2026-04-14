"""
Fraud Detection Model Training Script with MLflow Tracking.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --data data/raw/FraudShield_Banking_Data.csv
"""

import argparse
import pickle
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "FraudShield_Banking_Data.csv"
MODELS_DIR = BASE_DIR / "models"


# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_data(path: str = str(DATA_PATH)) -> pd.DataFrame:
    """Load and perform basic cleaning of raw data."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw shape: {df.shape}")

    # Drop rows with null Fraud_Label
    df = df.dropna(subset=["Fraud_Label"])

    # Drop ID columns - no predictive value
    drop_cols = ["Transaction_ID", "Customer_ID", "Merchant_ID", "Device_ID", "IP_Address"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    logger.info(f"After cleaning: {df.shape}")
    logger.info(f"Fraud distribution:\n{df['Fraud_Label'].value_counts()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    logger.info("Engineering features...")

    # Parse time → hour
    df["Transaction_Hour"] = df["Transaction_Time"].str.split(":").str[0].astype(float)

    # Parse date → day of week, month
    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Transaction_DayOfWeek"] = df["Transaction_Date"].dt.dayofweek
    df["Transaction_Month"] = df["Transaction_Date"].dt.month

    # Is same location as home?
    df["Is_Same_Location"] = (
        df["Transaction_Location"] == df["Customer_Home_Location"]
    ).astype(int)

    # Ratio features
    df["Amount_To_Balance_Ratio"] = df["Transaction_Amount (in Million)"] / (
        df["Account_Balance (in Million)"].replace(0, np.nan)
    ).fillna(0)

    df["Amount_To_Avg_Ratio"] = df["Transaction_Amount (in Million)"] / (
        df["Avg_Transaction_Amount (in Million)"].replace(0, np.nan)
    ).fillna(0)

    # Encode binary Yes/No columns
    for col in ["Is_International_Transaction", "Is_New_Merchant", "Unusual_Time_Transaction"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Encode target
    df["Fraud_Label"] = df["Fraud_Label"].map({"Fraud": 1, "Normal": 0})

    # Drop raw columns no longer needed
    df = df.drop(columns=["Transaction_Time", "Transaction_Date",
                           "Transaction_Location", "Customer_Home_Location"], errors="ignore")

    return df


def get_feature_columns():
    """Define categorical and numerical feature columns."""
    categorical = ["Transaction_Type", "Merchant_Category", "Card_Type"]
    numerical = [
        "Transaction_Amount (in Million)", "Transaction_Hour",
        "Transaction_DayOfWeek", "Transaction_Month",
        "Distance_From_Home", "Account_Balance (in Million)",
        "Daily_Transaction_Count", "Weekly_Transaction_Count",
        "Avg_Transaction_Amount (in Million)", "Max_Transaction_Last_24h (in Million)",
        "Is_International_Transaction", "Is_New_Merchant",
        "Failed_Transaction_Count", "Unusual_Time_Transaction",
        "Previous_Fraud_Count", "Is_Same_Location",
        "Amount_To_Balance_Ratio", "Amount_To_Avg_Ratio",
    ]
    return categorical, numerical


def build_pipeline(categorical: list, numerical: list) -> Pipeline:
    """Build sklearn preprocessing pipeline."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return preprocessor


# =============================================================================
# Model Training
# =============================================================================

def train_baseline(X_train, y_train, X_test, y_test, pipeline):
    """Train Logistic Regression baseline."""
    logger.info("Training Logistic Regression baseline...")

    with mlflow.start_run(run_name="LogisticRegression_Baseline", nested=True):
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(pipeline.transform(X_train), y_train)

        X_test_t = pipeline.transform(X_test)
        y_pred = lr.predict(X_test_t)
        y_proba = lr.predict_proba(X_test_t)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr, "model")

        logger.info(f"LR — AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")
        return lr, metrics


def train_xgboost(X_train, y_train, X_test, y_test, pipeline):
    """Train XGBoost with class imbalance handling."""
    logger.info("Training XGBoost...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="XGBoost_SMOTE", nested=True):
        # Apply SMOTE on training data
        X_train_t = pipeline.transform(X_train)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_t, y_train)
        logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

        xgb = XGBClassifier(**params)
        xgb.fit(
            X_resampled, y_resampled,
            eval_set=[(pipeline.transform(X_test), y_test)],
            verbose=False,
        )

        X_test_t = pipeline.transform(X_test)
        y_pred = xgb.predict(X_test_t)
        y_proba = xgb.predict_proba(X_test_t)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(xgb, "model")

        logger.info(f"XGBoost — AUC-ROC: {metrics['auc_roc']:.4f}, F1: {metrics['f1']:.4f}")
        return xgb, metrics


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute evaluation metrics."""
    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
    }


# =============================================================================
# Main
# =============================================================================

def main(data_path: str = str(DATA_PATH)):
    print("=" * 60)
    print("FraudShield Model Training")
    print("=" * 60)

    MODELS_DIR.mkdir(exist_ok=True)

    # Setup MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("fraud-detection")

    # Load & preprocess
    df = load_data(data_path)
    df = engineer_features(df)

    categorical, numerical = get_feature_columns()

    # Filter to available columns
    available_cols = [c for c in categorical + numerical if c in df.columns]
    categorical = [c for c in categorical if c in df.columns]
    numerical = [c for c in numerical if c in df.columns]

    X = df[categorical + numerical].copy()
    y = df["Fraud_Label"].copy()

    # Drop remaining nulls
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    logger.info(f"Final dataset: {X.shape}, Fraud rate: {y.mean():.4f}")

    # Train/test split — stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    preprocessor = build_pipeline(categorical, numerical)
    preprocessor.fit(X_train)

    feature_names = (
        numerical +
        list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical))
    )

    with mlflow.start_run(run_name="FraudShield_Training"):
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("fraud_rate", round(float(y.mean()), 4))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train models
        lr_model, lr_metrics = train_baseline(X_train, y_train, X_test, y_test, preprocessor)
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, preprocessor)

        # Best model = XGBoost
        best_model = xgb_model
        best_metrics = xgb_metrics
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

        # Print comparison
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(f"{'Metric':<20} {'Logistic Reg':>15} {'XGBoost':>15}")
        print("-" * 50)
        for k in lr_metrics:
            print(f"{k:<20} {lr_metrics[k]:>15.4f} {xgb_metrics[k]:>15.4f}")

        # Detailed XGBoost report
        X_test_t = preprocessor.transform(X_test)
        y_pred = best_model.predict(X_test_t)
        print("\nXGBoost Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

        # Save model + pipeline
        model_path = MODELS_DIR / "fraud_model.pkl"
        pipeline_path = MODELS_DIR / "pipeline.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        with open(pipeline_path, "wb") as f:
            pickle.dump({
                "pipeline": preprocessor,
                "feature_names": feature_names,
                "categorical": categorical,
                "numerical": numerical,
            }, f)

        print(f"\nModel saved to {model_path}")
        print(f"Pipeline saved to {pipeline_path}")

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(pipeline_path))

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best AUC-ROC: {best_metrics['auc_roc']:.4f}")
    print(f"Best F1:      {best_metrics['f1']:.4f}")
    print(f"Best Recall:  {best_metrics['recall']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()
    main(args.data)