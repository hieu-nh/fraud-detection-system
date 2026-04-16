"""
FraudShield Credit Card Fraud Detection — Train, Tune & Save Best Model.

Dataset: fraudTrain.csv / fraudTest.csv (comma-separated)
Train/Test already split — no need to split manually.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --skip-tune
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
TRAIN_PATH = BASE_DIR / "data" / "raw" / "fraudTrain.csv"
TEST_PATH  = BASE_DIR / "data" / "raw" / "fraudTest.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    'amt',
    'city_pop',
    'age',
    'hour',
    'day_of_week',
    'month',
    'is_night',
    'is_weekend',
    'distance',
    'amt_vs_category_mean',
    'category_enc',
    'gender_enc',
    'state_enc',
]


# =============================================================================
# Data Loading & Feature Engineering
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two coordinates."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame, label_encoders: dict = None, fit: bool = True):
    """
    Feature engineering for credit card fraud dataset.
    fit=True  → fit encoders on train set
    fit=False → use existing encoders for test set
    """
    df = df.copy()
    if label_encoders is None:
        label_encoders = {}

    # Datetime features
    dt = pd.to_datetime(df['trans_date_trans_time'], dayfirst=False, errors='coerce')
    df['hour']        = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['month']       = dt.dt.month
    df['is_night']    = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)

    # Age from dob
    dob = pd.to_datetime(df['dob'], dayfirst=True, errors='coerce')
    df['age'] = (dt - dob).dt.days / 365.25
    df['age'] = df['age'].fillna(df['age'].median()).clip(lower=0)

    # Distance cardholder → merchant (haversine)
    for col in ['lat', 'long', 'merch_lat', 'merch_long']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['distance'] = haversine(
        df['lat'].fillna(0),      df['long'].fillna(0),
        df['merch_lat'].fillna(0), df['merch_long'].fillna(0)
    )

    # Amount vs category mean
    if fit:
        cat_mean = df.groupby('category')['amt'].mean()
        label_encoders['cat_mean'] = cat_mean.to_dict()
    df['amt_vs_category_mean'] = (
        df['amt'] / df['category'].map(label_encoders['cat_mean']).fillna(1)
    )

    # Gender encode
    df['gender_enc'] = (df['gender'] == 'M').astype(int)

    # LabelEncoder for category and state
    for col, key in [('category', 'category'), ('state', 'state')]:
        if fit:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].fillna('Unknown'))
            label_encoders[key] = le
        else:
            le = label_encoders[key]
            df[f'{col}_enc'] = df[col].map(
                dict(zip(le.classes_, le.transform(le.classes_)))
            ).fillna(0).astype(int)

    return df, label_encoders


def load_and_prepare(train_path=str(TRAIN_PATH), test_path=str(TEST_PATH)):
    """Load both CSVs and apply feature engineering."""
    logger.info(f"Loading train: {train_path}")
    train = pd.read_csv(train_path, index_col=0)
    logger.info(f"Loading test:  {test_path}")
    test  = pd.read_csv(test_path,  index_col=0)

    logger.info(f"Train: {train.shape} | Fraud rate: {train['is_fraud'].mean():.4f}")
    logger.info(f"Test:  {test.shape}  | Fraud rate: {test['is_fraud'].mean():.4f}")

    # Engineer features — fit on train, transform on test
    train, label_encoders = engineer_features(train, fit=True)
    test,  _              = engineer_features(test, label_encoders=label_encoders, fit=False)

    available = [f for f in FEATURES if f in train.columns]

    X_train = train[available].fillna(0)
    y_train = train['is_fraud']
    X_test  = test[available].fillna(0)
    y_test  = test['is_fraud']

    logger.info(f"Features: {available}")
    return X_train, X_test, y_train, y_test, label_encoders


# =============================================================================
# Helpers
# =============================================================================

def find_best_threshold(y_test, y_proba):
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
    print("\n" + "=" * 95)
    print("Model Comparison (sorted by PR-AUC)")
    print("=" * 95)
    print(f"  {'Model':<35} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>7} {'Precision':>10} {'Recall':>8} {'Threshold':>10}")
    print("-" * 95)
    for name, m in sorted(results.items(), key=lambda x: x[1]['pr_auc'], reverse=True):
        print(f"  {name:<35} {m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f} "
              f"{m['f1']:>7.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['threshold']:>10.3f}")
    print("=" * 95)


# =============================================================================
# Step 1: Compare Models
# =============================================================================

def compare_models(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 60)
    print("Step 1: Model Comparison")
    print("=" * 60)

    # Scale for Logistic Regression
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Manual oversampling: fraud × 10
    fraud_idx = y_train[y_train == 1].index
    X_over    = pd.concat([X_train] + [X_train.loc[fraud_idx]] * 10)
    y_over    = pd.concat([y_train] + [y_train.loc[fraud_idx]] * 10)
    logger.info(f"Oversampled: {X_over.shape} | Fraud rate: {y_over.mean():.3f}")

    models_cfg = {
        "LogisticRegression_balanced": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5, random_state=42),
            "X_tr": X_train_sc, "X_te": X_test_sc, "y_tr": y_train,
            "scaler": scaler,
        },
        "RandomForest_balanced": {
            "model": RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                            max_depth=8, random_state=42, n_jobs=-1),
            "X_tr": X_train.values, "X_te": X_test.values, "y_tr": y_train,
            "scaler": None,
        },
        "GBM_oversampled": {
            "model": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=42),
            "X_tr": X_over.values, "X_te": X_test.values, "y_tr": y_over,
            "scaler": None,
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
                                 "proba": y_proba, "scaler": cfg["scaler"]}

                mlflow.log_params({"model_type": name, "threshold": best_thr})
                mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    print_comparison_table({k: v for k, v in results.items()})
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    print(f"\n🏆 Best by PR-AUC: {best_name}")
    return results, best_name


# =============================================================================
# Step 2: Tune Best Model
# =============================================================================

def tune_model(best_name, X_train, y_train, X_test, y_test, n_iter=30):
    print("\n" + "=" * 60)
    print(f"Step 2: Hyperparameter Tuning — {best_name} ({n_iter} iterations)")
    print("=" * 60)

    fraud_idx = y_train[y_train == 1].index
    X_over    = pd.concat([X_train] + [X_train.loc[fraud_idx]] * 10)
    y_over    = pd.concat([y_train] + [y_train.loc[fraud_idx]] * 10)

    final_scaler = None

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
        X_te = X_test.values

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
        X_te = X_test.values

    else:  # LogisticRegression
        param_dist = {
            "C":        [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "solver":   ["lbfgs", "liblinear"],
            "max_iter": [500, 1000, 2000],
        }
        final_scaler = StandardScaler()
        X_tr = final_scaler.fit_transform(X_train)
        y_tr = y_train
        X_te = final_scaler.transform(X_test)
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

    return tuned_model, metrics, final_scaler


# =============================================================================
# Step 3: Save
# =============================================================================

def save_model(model, label_encoders, metrics, threshold, model_name, scaler=None):
    MODELS_DIR.mkdir(exist_ok=True)

    with open(MODELS_DIR / "fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(MODELS_DIR / "pipeline.pkl", "wb") as f:
        pickle.dump({
            "label_encoders": label_encoders,
            "feature_names":  FEATURES,
            "best_threshold": threshold,
            "model_name":     model_name,
            "scaler":         scaler,
        }, f)

    print(f"\n✅ Saved fraud_model.pkl  → {MODELS_DIR / 'fraud_model.pkl'}")
    print(f"✅ Saved pipeline.pkl     → {MODELS_DIR / 'pipeline.pkl'}")
    print(f"\nFinal Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} = {v:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      default=str(TRAIN_PATH))
    parser.add_argument("--test",       default=str(TEST_PATH))
    parser.add_argument("--n_iter",     type=int, default=30)
    parser.add_argument("--skip-tune",  action="store_false", default=True)
    args = parser.parse_args()

    print("=" * 60)
    print("FraudShield Credit Card Fraud Detection — Training")
    print("=" * 60)
    print(f"n_iter:    {args.n_iter}")
    print(f"skip_tune: {args.skip_tune}")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("fraud-detection")

    # Load & prepare
    X_train, X_test, y_train, y_test, label_encoders = load_and_prepare(args.train, args.test)

    with mlflow.start_run(run_name="FraudShield_Training"):
        mlflow.log_params({
            "train_size":  len(X_train),
            "test_size":   len(X_test),
            "fraud_rate":  round(float(y_train.mean()), 4),
            "features":    len(FEATURES),
            "skip_tune":   args.skip_tune,
        })

        # Step 1: Compare
        results, best_name = compare_models(X_train, X_test, y_train, y_test)

        # Step 2: Tune or skip (default: skip)
        if args.skip_tune:
            print(f"\n⏭  Skipping tuning — using best from comparison: {best_name}")
            best_result     = results[best_name]
            final_model     = best_result["model"]
            final_metrics   = {k: v for k, v in best_result.items() if isinstance(v, (int, float))}
            final_threshold = best_result["threshold"]
            final_scaler    = best_result.get("scaler")
        else:
            final_model, final_metrics, final_scaler = tune_model(
                best_name, X_train, y_train, X_test, y_test, n_iter=args.n_iter
            )
            final_threshold = final_metrics["threshold"]
            best_name = f"{best_name}_tuned"

        mlflow.log_metrics({f"final_{k}": v for k, v in final_metrics.items()
                            if isinstance(v, (int, float))})

    # Step 3: Save
    save_model(final_model, label_encoders, final_metrics,
               final_threshold, best_name, scaler=final_scaler)

    # Classification report
    X_te = X_test.values if hasattr(X_test, 'values') else X_test
    if final_scaler is not None:
        X_te = final_scaler.transform(X_te)
    y_pred = (final_model.predict_proba(X_te)[:, 1] >= final_threshold).astype(int)
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