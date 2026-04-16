"""
Model Evaluation Script for FraudShield Fraud Detection.

Usage:
    python scripts/evaluate_model.py
"""

import pickle
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def load_artifacts():
    with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "pipeline.pkl", "rb") as f:
        pipeline_data = pickle.load(f)
    return model, pipeline_data


def load_test_data(pipeline_data):
    """Load and prepare test set using same pipeline as training."""
    import sys
    sys.path.append(str(BASE_DIR))
    from scripts.train_model import load_and_prepare

    _, X_test, _, y_test, _ = load_and_prepare()

    scaler = pipeline_data.get("scaler")
    X_arr  = X_test.values if hasattr(X_test, 'values') else X_test
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    return X_arr, y_test


def print_classification_report(y_test, y_pred, y_proba, threshold):
    print("\n" + "=" * 60)
    print(f"Classification Report (threshold={threshold:.2f})")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"AUC-ROC:           {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
    print(f"F1 Score:          {f1_score(y_test, y_pred):.4f}")
    print(f"Precision:         {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(y_test, y_pred):.4f}")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fraud"])
    ax.set_yticklabels(["Normal", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center",
                    color="white" if val > cm.max() / 2 else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:   {tn:,}")
    print(f"  False Positives:  {fp:,}")
    print(f"  False Negatives:  {fn:,}  ← missed fraud")
    print(f"  True Positives:   {tp:,}  ← caught fraud")
    logger.info("Saved confusion_matrix.png")


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — FraudShield GBM")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved roc_curve.png")


def plot_precision_recall_curve(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = f1_scores.argmax()
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.scatter(recall[best_idx], precision[best_idx], color="red", s=100, zorder=5,
                label=f"Best threshold = {best_thr:.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — FraudShield GBM")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved precision_recall_curve.png")

    print(f"\nOptimal Threshold Analysis:")
    print(f"  Best threshold: {best_thr:.3f}")
    print(f"  Precision:      {precision[best_idx]:.4f}")
    print(f"  Recall:         {recall[best_idx]:.4f}")
    print(f"  F1:             {f1_scores[best_idx]:.4f}")


def plot_feature_importance(model, feature_names, top_n=13):
    if not hasattr(model, 'feature_importances_'):
        logger.info("Model does not have feature_importances_, skipping")
        return

    importance = model.feature_importances_
    indices    = np.argsort(importance)[::-1][:top_n]
    top_names  = [feature_names[i] for i in indices]
    top_values = importance[indices]

    plt.figure(figsize=(10, 7))
    colors = ["#e74c3c" if v > top_values.mean() else "#3498db" for v in top_values]
    plt.barh(range(top_n), top_values[::-1], color=colors[::-1], edgecolor="black", alpha=0.8)
    plt.yticks(range(top_n), top_names[::-1])
    plt.xlabel("Feature Importance (gain)")
    plt.title(f"Top {top_n} Feature Importances — GBM")
    plt.axvline(x=top_values.mean(), color="orange", linestyle="--",
                label=f"Mean = {top_values.mean():.4f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature_importance.png")

    print(f"\nTop Features by Importance:")
    print("-" * 40)
    for name, val in zip(top_names[:10], top_values[:10]):
        print(f"  {name:<30} {val:.4f}")


def threshold_analysis(y_test, y_proba):
    print("\nThreshold Sensitivity Analysis:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged%':>10}")
    print("-" * 50)
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        y_pred = (y_proba >= thr).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        r = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        flagged = y_pred.mean() * 100
        print(f"  {thr:>8.2f} {p:>10.3f} {r:>8.3f} {f1:>8.3f} {flagged:>9.2f}%")


def main():
    print("=" * 60)
    print("FraudShield Model Evaluation")
    print("=" * 60)

    REPORTS_DIR.mkdir(exist_ok=True)

    model, pipeline_data = load_artifacts()
    feature_names = pipeline_data.get("feature_names", [])
    threshold     = pipeline_data.get("best_threshold", 0.5)

    logger.info("Loading test data...")
    X_test, y_test = load_test_data(pipeline_data)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    print_classification_report(y_test, y_pred, y_proba, threshold)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_feature_importance(model, feature_names)
    threshold_analysis(y_test, y_proba)

    print(f"\n✅ Reports saved to {REPORTS_DIR}/")
    print("   • confusion_matrix.png")
    print("   • roc_curve.png")
    print("   • precision_recall_curve.png")
    print("   • feature_importance.png")


if __name__ == "__main__":
    main()