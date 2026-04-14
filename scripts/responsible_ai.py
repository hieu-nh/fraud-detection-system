"""
Responsible AI Analysis for FraudShield Fraud Detection.

This script performs:
1. Model explainability with SHAP
2. Fairness analysis across demographic groups
3. Bias detection and reporting

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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "raw" / "FraudShield_Banking_Data.csv"
OUTPUT_DIR = BASE_DIR / "reports"


def load_artifacts():
    """Load trained model and preprocessing pipeline."""
    with open(MODELS_DIR / "fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "pipeline.pkl", "rb") as f:
        pipeline_data = pickle.load(f)
    return model, pipeline_data


def load_and_prepare_data():
    """Load and preprocess data for analysis."""
    import sys
    sys.path.append(str(BASE_DIR))
    from scripts.train_model import load_data, engineer_features, get_feature_columns

    df = load_data()
    df_orig = pd.read_csv(DATA_PATH).dropna(subset=["Fraud_Label"])
    df = engineer_features(df)

    categorical, numerical = get_feature_columns()
    categorical = [c for c in categorical if c in df.columns]
    numerical = [c for c in numerical if c in df.columns]

    X = df[categorical + numerical].copy()
    y = df["Fraud_Label"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    # Keep original columns for fairness analysis
    df_orig = df_orig.loc[mask.index[mask]].reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y, df_orig, categorical, numerical


# =============================================================================
# 1. SHAP Explainability
# =============================================================================

def explain_with_shap(model, X_transformed, feature_names, n_samples=500):
    """Generate SHAP explanations."""
    logger.info("Computing SHAP values...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Sample for efficiency
    idx = np.random.choice(len(X_transformed), min(n_samples, len(X_transformed)), replace=False)
    X_sample = X_transformed[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, use fraud class (index 1)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # --- Plot 1: Summary (bar) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — Top Features for Fraud Detection")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_importance.png")

    # --- Plot 2: Beeswarm ---
    plt.figure(figsize=(10, 10))
    shap.summary_plot(sv, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Value Distribution — Impact on Fraud Probability")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved shap_beeswarm.png")

    # Top features
    mean_abs = np.abs(sv).mean(axis=0)
    top_features = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)[:10]

    print("\n📊 Top 10 Most Important Features (SHAP):")
    print("-" * 45)
    for feat, val in top_features:
        print(f"  {feat:<40} {val:.4f}")

    return sv, top_features


# =============================================================================
# 2. Fairness Analysis
# =============================================================================

def fairness_analysis(model, pipeline, X, y, df_orig):
    """
    Analyze model fairness across demographic/geographic groups.

    Checks:
    - Equal opportunity (recall per group)
    - Predictive parity (precision per group)
    - Demographic parity (prediction rate per group)
    """
    logger.info("Running fairness analysis...")

    OUTPUT_DIR.mkdir(exist_ok=True)

    X_transformed = pipeline["pipeline"].transform(X)
    y_pred = model.predict(X_transformed)
    y_proba = model.predict_proba(X_transformed)[:, 1]

    results = {}

    # --- Group 1: Card Type ---
    if "Card_Type" in df_orig.columns:
        results["Card_Type"] = _group_metrics(df_orig["Card_Type"], y, y_pred, y_proba)

    # --- Group 2: Transaction Type ---
    if "Transaction_Type" in df_orig.columns:
        results["Transaction_Type"] = _group_metrics(
            df_orig["Transaction_Type"], y, y_pred, y_proba
        )

    # --- Group 3: International vs Domestic ---
    if "Is_International_Transaction" in df_orig.columns:
        results["Is_International"] = _group_metrics(
            df_orig["Is_International_Transaction"], y, y_pred, y_proba
        )

    # Print fairness report
    print("\n⚖️  Fairness Analysis Report")
    print("=" * 70)
    for group_name, group_results in results.items():
        print(f"\n📌 Group: {group_name}")
        print(f"{'Subgroup':<25} {'Recall':>8} {'Precision':>10} {'Pred Rate':>10} {'AUC':>8} {'N':>6}")
        print("-" * 70)
        for subgroup, metrics in group_results.items():
            print(
                f"  {str(subgroup):<23} "
                f"{metrics['recall']:>8.3f} "
                f"{metrics['precision']:>10.3f} "
                f"{metrics['pred_rate']:>10.3f} "
                f"{metrics['auc']:>8.3f} "
                f"{metrics['n']:>6}"
            )

    # Check for disparate impact
    print("\n🔍 Disparity Detection:")
    for group_name, group_results in results.items():
        recalls = [m["recall"] for m in group_results.values()]
        if max(recalls) - min(recalls) > 0.1:
            print(f"  ⚠️  {group_name}: Recall disparity = {max(recalls) - min(recalls):.3f} (>0.10 threshold)")
        else:
            print(f"  ✅ {group_name}: Recall disparity = {max(recalls) - min(recalls):.3f} (within threshold)")

    # Save fairness report
    _save_fairness_chart(results)

    return results


def _group_metrics(group_series, y_true, y_pred, y_proba):
    """Compute metrics for each subgroup."""
    metrics = {}
    for subgroup in group_series.dropna().unique():
        mask = group_series == subgroup
        if mask.sum() < 10:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_proba[mask]

        if len(yt.unique()) < 2:
            continue

        metrics[subgroup] = {
            "n": int(mask.sum()),
            "recall": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
            "precision": float(
                len(yt[(yp == 1) & (yt == 1)]) / max(1, (yp == 1).sum())
            ),
            "pred_rate": float((yp == 1).mean()),
            "auc": float(roc_auc_score(yt, ypr)),
        }
    return metrics


def _save_fairness_chart(results):
    """Save fairness comparison bar chart."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (group_name, group_results) in zip(axes, results.items()):
        subgroups = list(group_results.keys())
        recalls = [group_results[s]["recall"] for s in subgroups]
        colors = ["#e74c3c" if r < 0.5 else "#2ecc71" for r in recalls]
        ax.bar(subgroups, recalls, color=colors, edgecolor="black", alpha=0.8)
        ax.set_title(f"Recall by {group_name}")
        ax.set_ylabel("Recall (Fraud Detection Rate)")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.8, color="orange", linestyle="--", label="0.8 target")
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle("Fairness Analysis — Recall Across Groups", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fairness_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved fairness_analysis.png")


# =============================================================================
# 3. Ethics Report
# =============================================================================

def print_ethics_report():
    """Print ethical considerations and mitigations."""
    print("\n🧭 Ethical Considerations & Mitigations")
    print("=" * 60)
    considerations = [
        ("Data Privacy",
         "Transaction data is sensitive PII. In production:\n"
         "   • Anonymize Customer_ID and Device_ID\n"
         "   • Store data encrypted at rest\n"
         "   • Apply data retention policies"),

        ("Algorithmic Bias",
         "Model may reflect historical bias in fraud patterns.\n"
         "   • Regular fairness audits across geographic regions\n"
         "   • Threshold adjustment per demographic group\n"
         "   • Human review for CRITICAL risk decisions"),

        ("False Positives Impact",
         "Wrongly blocking legitimate transactions harms customers.\n"
         "   • Set threshold conservatively (0.5 default)\n"
         "   • Provide appeal/dispute mechanism\n"
         "   • Monitor false positive rate per customer segment"),

        ("Model Drift",
         "Fraud patterns evolve over time.\n"
         "   • Monitor prediction distribution weekly\n"
         "   • Retrain monthly with fresh data\n"
         "   • Alert on AUC-ROC degradation > 5%"),

        ("Transparency",
         "Customers should understand why transactions are flagged.\n"
         "   • SHAP explanations for each decision\n"
         "   • Provide top-3 risk factors to customer service\n"
         "   • Maintain audit log of all flagged transactions"),
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

    model, pipeline_data = load_artifacts()
    X, y, df_orig, categorical, numerical = load_and_prepare_data()

    pipeline = pipeline_data["pipeline"]
    feature_names = pipeline_data["feature_names"]
    X_transformed = pipeline.transform(X)

    # 1. SHAP
    print("\n[1/3] SHAP Explainability...")
    shap_values, top_features = explain_with_shap(model, X_transformed, feature_names)

    # 2. Fairness
    print("\n[2/3] Fairness Analysis...")
    fairness_results = fairness_analysis(model, pipeline_data, X, y, df_orig)

    # 3. Ethics
    print("\n[3/3] Ethics Report...")
    print_ethics_report()

    print(f"\n✅ Reports saved to {OUTPUT_DIR}/")
    print("   • shap_importance.png")
    print("   • shap_beeswarm.png")
    print("   • fairness_analysis.png")


if __name__ == "__main__":
    main()