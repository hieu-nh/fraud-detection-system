"""
Responsible AI Analysis for FraudShield Fraud Detection.

1. SHAP explainability
2. Fairness analysis across gender, category, state
3. Ethics report

Usage:
    python scripts/responsible_ai.py
"""

import pickle
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent.parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


def load_artifacts():
    with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "pipeline.pkl", "rb") as f:
        pipeline_data = pickle.load(f)
    return model, pipeline_data


def load_data_for_analysis():
    """Load train/test with both engineered features and original columns."""
    import sys
    sys.path.append(str(BASE_DIR))
    from scripts.train_model import load_and_prepare, engineer_features, FEATURES

    # Load raw test for fairness groups
    from scripts.train_model import TEST_PATH
    test_raw = pd.read_csv(TEST_PATH, index_col=0)

    # Get engineered test features
    _, X_test, _, y_test, pipeline_data_local = load_and_prepare()

    return X_test, y_test, test_raw, pipeline_data_local


# =============================================================================
# 1. SHAP Explainability
# =============================================================================

def explain_with_shap(model, X, feature_names, n_samples=500):
    logger.info("Computing SHAP values...")
    REPORTS_DIR.mkdir(exist_ok=True)

    X_arr = X.values if hasattr(X, 'values') else X
    idx   = np.random.choice(len(X_arr), min(n_samples, len(X_arr)), replace=False)
    X_sample = X_arr[idx]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Bar plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — FraudShield GBM")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_importance.png")

    # Beeswarm plot
    plt.figure(figsize=(10, 9))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Value Distribution — Impact on Fraud Probability")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_beeswarm.png")

    mean_abs     = np.abs(sv).mean(axis=0)
    top_features = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)[:10]

    print("\n📊 Top 10 Most Important Features (SHAP):")
    print("-" * 45)
    for feat, val in top_features:
        print(f"  {feat:<35} {val:.4f}")

    return sv, top_features


# =============================================================================
# 2. Fairness Analysis
# =============================================================================

def fairness_analysis(model, X_test, y_test, test_raw, pipeline_data):
    logger.info("Running fairness analysis...")
    REPORTS_DIR.mkdir(exist_ok=True)

    scaler = pipeline_data.get("scaler")
    X_arr  = X_test.values if hasattr(X_test, 'values') else X_test
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    y_proba = model.predict_proba(X_arr)[:, 1]
    threshold = pipeline_data.get("best_threshold", 0.5)
    y_pred  = (y_proba >= threshold).astype(int)

    # Align index
    test_raw = test_raw.iloc[:len(y_test)].copy()
    test_raw = test_raw.reset_index(drop=True)
    y_test   = y_test.reset_index(drop=True)
    y_pred_s = pd.Series(y_pred)
    y_proba_s = pd.Series(y_proba)

    results = {}

    for group_col in ['gender', 'category', 'state']:
        if group_col not in test_raw.columns:
            continue
        results[group_col] = _group_metrics(
            test_raw[group_col].reset_index(drop=True),
            y_test, y_pred_s, y_proba_s
        )

    # Print report
    print("\n⚖️  Fairness Analysis Report")
    print("=" * 70)
    for group_name, group_results in results.items():
        print(f"\n📌 Group: {group_name}")
        print(f"  {'Subgroup':<20} {'Recall':>8} {'Precision':>10} {'Pred Rate':>10} {'AUC':>8} {'N':>8}")
        print("-" * 70)
        for subgroup, m in sorted(group_results.items(), key=lambda x: x[1]['recall'], reverse=True):
            print(f"  {str(subgroup):<20} {m['recall']:>8.3f} {m['precision']:>10.3f} "
                  f"{m['pred_rate']:>10.3f} {m['auc']:>8.3f} {m['n']:>8}")

    # Disparity check
    print("\n🔍 Disparity Detection:")
    for group_name, group_results in results.items():
        recalls = [m['recall'] for m in group_results.values()]
        if len(recalls) < 2:
            continue
        disparity = max(recalls) - min(recalls)
        flag = "⚠️ " if disparity > 0.10 else "✅"
        print(f"  {flag} {group_name}: Recall disparity = {disparity:.3f}")

    _save_fairness_chart(results)
    return results


def _group_metrics(group_series, y_true, y_pred, y_proba):
    metrics = {}
    for subgroup in group_series.dropna().unique():
        mask = (group_series == subgroup).values
        if mask.sum() < 20:
            continue
        yt  = y_true[mask]
        yp  = y_pred[mask]
        ypr = y_proba[mask]
        if len(yt.unique()) < 2:
            continue
        tp = ((yp == 1) & (yt == 1)).sum()
        fp = ((yp == 1) & (yt == 0)).sum()
        fn = ((yp == 0) & (yt == 1)).sum()
        metrics[subgroup] = {
            "n":          int(mask.sum()),
            "recall":     float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            "precision":  float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            "pred_rate":  float(yp.mean()),
            "auc":        float(roc_auc_score(yt, ypr)) if len(yt.unique()) > 1 else 0,
        }
    return metrics


def _save_fairness_chart(results):
    n_groups = len(results)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
    if n_groups == 1:
        axes = [axes]

    for ax, (group_name, group_results) in zip(axes, results.items()):
        subgroups = list(group_results.keys())[:10]  # max 10
        recalls   = [group_results[s]['recall'] for s in subgroups]
        colors    = ["#e74c3c" if r < 0.5 else "#2ecc71" for r in recalls]
        ax.bar(range(len(subgroups)), recalls, color=colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(subgroups)))
        ax.set_xticklabels(subgroups, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Recall by {group_name}")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.7, color="orange", linestyle="--", label="0.7 target")
        ax.legend()

    plt.suptitle("Fairness Analysis — Recall Across Groups", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved fairness_analysis.png")


# =============================================================================
# 3. Ethics Report
# =============================================================================

def print_ethics_report():
    print("\n🧭 Ethical Considerations & Mitigations")
    print("=" * 60)
    considerations = [
        ("Data Privacy",
         "Transaction data contains PII (name, location, card number).\n"
         "   • Anonymize cc_num, first, last before storage\n"
         "   • Apply data retention policies\n"
         "   • Encrypt sensitive fields at rest"),

        ("Algorithmic Bias",
         "Model trained on historical data may reflect past biases.\n"
         "   • Regular fairness audits across gender, state, category\n"
         "   • Monitor recall parity — all groups should have similar detection rates\n"
         "   • Human review required for CRITICAL risk decisions"),

        ("False Positives Impact",
         "Wrongly blocking legitimate transactions harms customers.\n"
         "   • Threshold set at 0.85 to minimize false positives\n"
         "   • Provide dispute mechanism for flagged transactions\n"
         "   • Monitor FP rate per customer segment weekly"),

        ("Model Drift",
         "Fraud patterns evolve — model degrades over time.\n"
         "   • Monitor PR-AUC weekly on live predictions\n"
         "   • Retrain monthly with fresh labeled data\n"
         "   • Alert when PR-AUC drops > 5% from baseline"),

        ("Transparency",
         "Customers deserve to know why transactions are flagged.\n"
         "   • SHAP top-3 features provided to customer service agents\n"
         "   • Maintain audit log of all flagged transactions\n"
         "   • Clear escalation path for disputed decisions"),
    ]
    for title, detail in considerations:
        print(f"\n  📌 {title}")
        print(f"     {detail}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("FraudShield Responsible AI Analysis")
    print("=" * 60)

    REPORTS_DIR.mkdir(exist_ok=True)

    model, pipeline_data = load_artifacts()
    feature_names = pipeline_data.get("feature_names", [])

    print("\n[1/3] Loading data...")
    X_test, y_test, test_raw, _ = load_data_for_analysis()

    # Apply scaler if needed
    scaler = pipeline_data.get("scaler")
    X_arr  = X_test.values if hasattr(X_test, 'values') else X_test
    if scaler is not None:
        X_arr = scaler.transform(X_arr)

    print("\n[2/3] SHAP Explainability...")
    shap_values, top_features = explain_with_shap(model, X_arr, feature_names)

    print("\n[3/3] Fairness Analysis...")
    fairness_results = fairness_analysis(model, X_test, y_test, test_raw, pipeline_data)

    print_ethics_report()

    print(f"\n✅ Reports saved to {REPORTS_DIR}/")
    print("   • shap_importance.png")
    print("   • shap_beeswarm.png")
    print("   • fairness_analysis.png")


if __name__ == "__main__":
    main()