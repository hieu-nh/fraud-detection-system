"""
FraudShield Model Training — Compare, Tune & Save Best Model.

Workflow:
  1. Load & engineer features from raw data
  2. Compare 3 models (LogisticRegression, RandomForest, GBM)
  3. Tune best model with RandomizedSearch
  4. Save best model + pipeline artifacts

All experiments tracked in MLflow.

Usage:
    # Full pipeline: compare → tune → save
    python scripts/train_model.py

    # Skip tuning (faster, uses default params)
    python scripts/train_model.py --skip-tune

    # Custom iterations
    python scripts/train_model.py --n_iter 50
"""

import argparse
import os
import pickle
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score,
    confusion_matrix, precision_score, recall_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "FraudShield_Banking_Data.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    'Transaction_Amount (in Million)',
    'Distance_From_Home',
    'Account_Balance (in Million)',
    'Daily_Transaction_Count',
    'Weekly_Transaction_Count',
    'Avg_Transaction_Amount (in Million)',
    'Max_Transaction_Last_24h (in Million)',
    'Failed_Transaction_Count',
    'Previous_Fraud_Count',
    'Is_International_Transaction',
    'Is_New_Merchant',
    'Unusual_Time_Transaction',
    'Transaction_Type_enc',
    'Merchant_Category_enc',
    'Card_Type_enc',
    'risk_score',
    'amount_vs_avg',
    'is_night',
    'is_weekend',
]


# =============================================================================
# Data Loading & Feature Engineering
# =============================================================================

def load_and_prepare(path: str = str(DATA_PATH)):
    """Load raw data and apply feature engineering."""
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=['Fraud_Label']).reset_index(drop=True)

    # Encode target
    df['Fraud_Label'] = (df['Fraud_Label'].str.strip() == 'Fraud').astype(int)

    # Time features
    df['hour']        = pd.to_datetime(df['Transaction_Time'], format='%H:%M', errors='coerce').dt.hour
    dates             = pd.to_datetime(df['Transaction_Date'], errors='coerce')
    df['day_of_week'] = dates.dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # Binary flags
    for col in ['Is_International_Transaction', 'Is_New_Merchant', 'Unusual_Time_Transaction']:
        df[col] = (df[col].str.strip() == 'Yes').astype(float)

    # LabelEncoder for categoricals
    label_encoders = {}
    for col in ['Transaction_Type', 'Merchant_Category', 'Card_Type']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].fillna('Unknown'))
        label_encoders[col] = le

    # Composite risk score
    df['risk_score'] = (
        df['Is_International_Transaction'] * 0.30 +
        df['Is_New_Merchant']              * 0.20 +
        df['Unusual_Time_Transaction']     * 0.25 +
        (df['Failed_Transaction_Count'].fillna(0) > 0).astype(float) * 0.15 +
        df['is_night']                     * 0.10
    ).round(3)

    # Amount vs average ratio
    df['amount_vs_avg'] = np.where(
        df['Avg_Transaction_Amount (in Million)'] > 0,
        df['Transaction_Amount (in Million)'] / df['Avg_Transaction_Amount (in Million)'],
        1.0
    )

    available = [f for f in FEATURES if f in df.columns]
    X = df[available].fillna(df[available].median(numeric_only=True))
    y = df['Fraud_Label']

    logger.info(f"Dataset: {X.shape} | Fraud rate: {y.mean():.4f}")
    return X, y, label_encoders


# =============================================================================
# Helpers
# =============================================================================

def find_best_threshold(y_test, y_proba):
    """Find threshold that maximizes F1 for fraud class."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1, best_thr = 0, 0.5
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr
    return best_thr, best_f1


def compute_metrics(y_test, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[1,1]; fp = cm[0,1]; fn = cm[1,0]
    return {
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc":    round(average_precision_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall":    round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
    }


def print_comparison_table(results: dict):
    print("\n" + "=" * 90)
    print("Model Comparison (sorted by PR-AUC)")
    print("=" * 90)
    print(f"  {'Model':<35} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>7} {'Precision':>10} {'Recall':>8} {'Threshold':>10}")
    print("-" * 90)
    for name, m in sorted(results.items(), key=lambda x: x[1]["pr_auc"], reverse=True):
        print(f"  {name:<35} {m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f} "
              f"{m['f1']:>7.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['threshold']:>10.3f}")
    print("=" * 90)


# =============================================================================
# Step 1: Compare models with default params
# =============================================================================

def compare_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Step 1: Model Comparison")
    print("=" * 60)

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    # Manual oversampling: fraud × 10
    fraud_idx = y_train[y_train == 1].index
    X_over    = pd.concat([X_train] + [X_train.loc[fraud_idx]] * 10)
    y_over    = pd.concat([y_train] + [y_train.loc[fraud_idx]] * 10)
    logger.info(f"Oversampled: {X_over.shape} | Fraud: {y_over.mean():.3f}")

    models_cfg = {
        "LogisticRegression_balanced": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5, random_state=42),
            "X_tr": X_train_sc, "X_te": X_test_sc, "y_tr": y_train,
        },
        "RandomForest_balanced": {
            "model": RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                            max_depth=8, random_state=42, n_jobs=-1),
            "X_tr": X_train.values, "X_te": X_test.values, "y_tr": y_train,
        },
        "GBM_oversampled": {
            "model": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=42),
            "X_tr": X_over.values, "X_te": X_test.values, "y_tr": y_over,
        },
    }

    results = {}

    with mlflow.start_run(run_name="Model_Comparison", nested=True):
        for name, cfg in models_cfg.items():
            logger.info(f"  Training {name}...")
            with mlflow.start_run(run_name=name, nested=True):
                clf = cfg["model"]
                clf.fit(cfg["X_tr"], cfg["y_tr"])

                y_proba = clf.predict_proba(cfg["X_te"])[:, 1]
                best_thr, _ = find_best_threshold(y_test, y_proba)
                y_pred  = (y_proba >= best_thr).astype(int)

                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics["threshold"] = round(float(best_thr), 3)
                results[name] = {**metrics, "model": clf,
                                 "X_te": cfg["X_te"], "proba": y_proba,
                                 "scaler": scaler if "Logistic" in name else None}

                mlflow.log_params({"model_type": name, "threshold": best_thr})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    print_comparison_table({k: v for k, v in results.items()})

    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\n🏆 Best by PR-AUC: {best_name}")
    return results, best_name


# =============================================================================
# Step 2: Tune best model
# =============================================================================

def tune_model(best_name, X_train, y_train, X_test, y_test, n_iter=30):
    print("\n" + "=" * 60)
    print(f"Step 2: Hyperparameter Tuning — {best_name} ({n_iter} iterations)")
    print("=" * 60)

    # Manual oversampling
    fraud_idx = y_train[y_train == 1].index
    X_over    = pd.concat([X_train] + [X_train.loc[fraud_idx]] * 10)
    y_over    = pd.concat([y_train] + [y_train.loc[fraud_idx]] * 10)

    if "GBM" in best_name:
        param_dist = {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [3, 4, 5, 6],
            "learning_rate":    [0.01, 0.05, 0.1, 0.15],
            "subsample":        [0.7, 0.8, 0.9, 1.0],
            "min_samples_leaf": [1, 5, 10, 20],
        }
        base = GradientBoostingClassifier(random_state=42)
        X_tr, y_tr = X_over.values, y_over

    elif "RandomForest" in best_name:
        param_dist = {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [4, 6, 8, 10, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", 0.5],
        }
        base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
        X_tr, y_tr = X_train.values, y_train

    else:  # LogisticRegression
        param_dist = {
            "C":        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver":   ["lbfgs", "liblinear"],
            "max_iter": [500, 1000, 2000],
        }
        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X_train)
        y_tr    = y_train
        X_test  = scaler.transform(X_test)
        base = LogisticRegression(class_weight='balanced', random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=n_iter,
        scoring="average_precision",
        cv=cv, n_jobs=-1, random_state=42, verbose=1,
    )

    with mlflow.start_run(run_name=f"{best_name}_Tuned", nested=True):
        search.fit(X_tr, y_tr)
        tuned_model = search.best_estimator_

        X_te = X_test.values if hasattr(X_test, 'values') else X_test
        y_proba = tuned_model.predict_proba(X_te)[:, 1]
        best_thr, _ = find_best_threshold(y_test, y_proba)
        y_pred  = (y_proba >= best_thr).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["threshold"] = round(float(best_thr), 3)

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_metric("best_cv_pr_auc", search.best_score_)
        mlflow.sklearn.log_model(tuned_model, "model")

        print(f"\n✅ Best CV PR-AUC: {search.best_score_:.4f}")
        print(f"Best Params: {search.best_params_}")

    return tuned_model, metrics


# =============================================================================
# Step 3: Save best model
# =============================================================================

def save_model(model, label_encoders, metrics, threshold, model_name):
    MODELS_DIR.mkdir(exist_ok=True)

    model_path    = MODELS_DIR / "fraud_model.pkl"
    pipeline_path = MODELS_DIR / "pipeline.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(pipeline_path, "wb") as f:
        pickle.dump({
            "label_encoders":  label_encoders,
            "feature_names":   FEATURES,
            "best_threshold":  threshold,
            "model_name":      model_name,
        }, f)

    print(f"\n✅ Saved fraud_model.pkl  → {model_path}")
    print(f"✅ Saved pipeline.pkl     → {pipeline_path}")
    print(f"\nFinal Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} = {v:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=str(DATA_PATH))
    parser.add_argument("--n_iter",     type=int, default=30)
    parser.add_argument("--skip-tune",  action="store_true",
                        help="Skip tuning, save best model from comparison directly")
    args = parser.parse_args()

    print("=" * 60)
    print("FraudShield Model Training")
    print("=" * 60)
    print(f"n_iter:     {args.n_iter}")
    print(f"skip_tune:  {args.skip_tune}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("fraud-detection")

    # Load data
    X, y, label_encoders = load_and_prepare(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    with mlflow.start_run(run_name="FraudShield_Training"):
        mlflow.log_params({
            "dataset_size": len(X),
            "fraud_rate":   round(float(y.mean()), 4),
            "train_size":   len(X_train),
            "test_size":    len(X_test),
            "skip_tune":    args.skip_tune,
        })

        # Step 1: Compare models
        results, best_name = compare_models(X_train, X_test, y_train, y_test)

        # Step 2: Tune or use best from comparison
        if args.skip_tune:
            print(f"\n⏭  Skipping tuning — saving best model from comparison: {best_name}")
            best_result = results[best_name]
            final_model = best_result["model"]
            final_metrics = {k: v for k, v in best_result.items()
                             if isinstance(v, (int, float))}
            final_threshold = best_result["threshold"]
        else:
            final_model, final_metrics = tune_model(
                best_name, X_train, y_train, X_test, y_test, n_iter=args.n_iter
            )
            final_threshold = final_metrics["threshold"]
            best_name = f"{best_name}_tuned"

        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()
                            if isinstance(v, (int, float))})

    # Step 3: Save
    save_model(final_model, label_encoders, final_metrics, final_threshold, best_name)

    # Print final classification report
    X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_pred = (final_model.predict_proba(X_test_arr)[:, 1] >= final_threshold).astype(int)
    print(f"\nClassification Report (threshold={final_threshold:.2f}):")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Best model:  {best_name}")
    print(f"  Threshold:   {final_threshold:.3f}")
    print(f"  PR-AUC:      {final_metrics.get('pr_auc', 'N/A')}")
    print(f"\nNext: python scripts/evaluate_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()