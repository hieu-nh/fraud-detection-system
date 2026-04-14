"""
Model Evaluation Script for FraudShield Fraud Detection.

Evaluates the trained model on test data and generates:
- Classification report
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Feature importance plot

Usage:
    python scripts/evaluate_model.py
"""

import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def load_artifacts():
    """Load trained model and preprocessing pipeline."""
    with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "pipeline.pkl", "rb") as f:
        pipeline_data = pickle.load(f)
    return model, pipeline_data


def load_test_data(pipeline_data):
    """Load and prepare test split."""
    import sys
    sys.path.append(str(BASE_DIR))
    from scripts.train_model import load_data, engineer_features, get_feature_columns

    df = load_data()
    df = engineer_features(df)

    categorical, numerical = get_feature_columns()
    categorical = [c for c in categorical if c in df.columns]
    numerical = [c for c in numerical if c in df.columns]

    X = df[categorical + numerical].copy()
    y = df["Fraud_Label"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    # Same split as training — same random_state gives same test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = pipeline_data["pipeline"]
    X_test_t = pipeline.transform(X_test)

    return X_test_t, y_test


# =============================================================================
# Evaluation Functions
# =============================================================================

def print_classification_report(y_test, y_pred, y_proba):
    """Print full classification metrics."""
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    print(f"AUC-ROC:           {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    print(f"F1 Score:          {f1_score(y_test, y_pred):.4f}")
    print(f"Precision:         {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(y_test, y_pred):.4f}")


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    ax.set_yticklabels(["Normal", "Fraud"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black", fontsize=14)

    plt.tight_layout()
    path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (Normal → Normal): {tn}")
    print(f"  False Positives (Normal → Fraud):  {fp}")
    print(f"  False Negatives (Fraud  → Normal): {fn}  ← missed fraud!")
    print(f"  True Positives  (Fraud  → Fraud):  {tp}  ← caught fraud!")


def plot_roc_curve(y_test, y_proba):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — FraudShield XGBoost")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")


def plot_precision_recall_curve(y_test, y_proba):
    """Plot and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # Find threshold closest to F1-optimal
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="blue", lw=2,
             label=f"PR curve (AP = {ap:.4f})")
    plt.scatter(recall[best_idx], precision[best_idx], color="red", s=100, zorder=5,
                label=f"Best threshold = {best_threshold:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — FraudShield XGBoost")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = REPORTS_DIR / "precision_recall_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")

    print(f"\nOptimal Threshold Analysis:")
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Precision at best: {precision[best_idx]:.4f}")
    print(f"  Recall at best:    {recall[best_idx]:.4f}")
    print(f"  F1 at best:        {f1_scores[best_idx]:.4f}")


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot XGBoost feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    top_names = [feature_names[i] for i in indices]
    top_values = importance[indices]

    plt.figure(figsize=(10, 8))
    colors = ["#e74c3c" if v > top_values.mean() else "#3498db" for v in top_values]
    plt.barh(range(top_n), top_values[::-1], color=colors[::-1], edgecolor="black", alpha=0.8)
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel("Feature Importance (gain)")
    plt.title(f"Top {top_n} Feature Importances — XGBoost")
    plt.axvline(x=top_values.mean(), color="orange", linestyle="--",
                label=f"Mean = {top_values.mean():.4f}")
    plt.legend()
    plt.tight_layout()

    path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {path}")

    print(f"\nTop 10 Features by Importance:")
    print("-" * 45)
    for name, val in zip(top_names[:10], top_values[:10]):
        print(f"  {name:<40} {val:.4f}")


def threshold_analysis(y_test, y_proba):
    """Show metrics at different thresholds."""
    print("\nThreshold Sensitivity Analysis:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged%':>10}")
    print("-" * 50)
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = (y_proba >= threshold).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        flagged = y_pred.mean() * 100
        print(f"  {threshold:>8.1f} {p:>10.3f} {r:>8.3f} {f1:>8.3f} {flagged:>9.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("FraudShield Model Evaluation")
    print("=" * 60)

    REPORTS_DIR.mkdir(exist_ok=True)

    # Load artifacts
    model, pipeline_data = load_artifacts()
    feature_names = pipeline_data["feature_names"]

    # Load test data
    logger.info("Loading test data...")
    X_test, y_test = load_test_data(pipeline_data)

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Reports
    print_classification_report(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, feature_names)
    threshold_analysis(y_test, y_proba)

    print(f"\n✅ All evaluation reports saved to {REPORTS_DIR}/")
    print("   • confusion_matrix.png")
    print("   • roc_curve.png")
    print("   • precision_recall_curve.png")
    print("   • feature_importance.png")


if __name__ == "__main__":
    main()