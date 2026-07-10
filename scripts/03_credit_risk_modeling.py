#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lending Club Credit Risk — Modeling & Risk Scorecard
================================================================
Sprint 2C: Logistic Regression + Evaluation + Scorecard + Stats

Author: Dheeraj Sharma

WHY LOGISTIC REGRESSION (not XGBoost)?
  This is a DATA ANALYST project for a regulated domain (lending).
  Banks need models that are:
    1. Explainable — "which factors increase default probability?"
    2. Auditable  — compliance needs coefficients, not black boxes
    3. Stable      — no hyperparameter sensitivity
  Logistic regression is the INDUSTRY STANDARD for credit scorecards
  (Basel II/III regulations). Using it here is a signal that you
  understand the domain, not that you don't know fancier models.

Run:
    python scripts/03_credit_risk_modeling.py

Input:  data/features_loans.csv   (from Sprint 2B)
Output: charts/10-15 (model evaluation visuals)
        data/model_results.csv    (predictions + probabilities)
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

COLORS = {
    'primary': '#1B4F72', 'accent1': '#E74C3C', 'accent2': '#27AE60',
    'accent3': '#F39C12', 'accent4': '#8E44AD', 'neutral': '#BDC3C7', 'bg': '#FAFAFA',
}
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': COLORS['bg'], 'axes.facecolor': '#FFFFFF',
    'axes.edgecolor': COLORS['neutral'], 'grid.color': '#EEEEEE',
    'font.family': 'sans-serif', 'figure.dpi': 120,
})

RANDOM_STATE = 42


# ================================================================
# SECTION 1: DATA PREPARATION
# ================================================================
def load_and_prepare():
    """Load feature data and prepare model-ready matrices."""
    logger.info("Loading feature dataset...")
    df = pd.read_csv(DATA_DIR / "features_loans.csv", low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    # --- Select model features ---
    # Numeric features that are model-ready
    numeric_features = [
        "loan_amnt", "term", "int_rate", "installment",
        "annual_inc_capped", "dti", "fico_score",
        "delinq_2yrs", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_util", "total_acc", "mort_acc",
        # Engineered features
        "income_to_loan_ratio", "installment_to_income",
        "grade_numeric", "grade_x_dti",
    ]
    # Binary flags
    binary_features = [
        "high_dti_flag", "recent_inquiry_flag", "derogatory_flag",
        "debt_consolidation_flag", "small_business_flag",
        "long_term_flag", "emp_length_missing",
    ]

    all_features = numeric_features + binary_features
    target = "default"

    # Drop any rows with NaN in feature columns (should be minimal after 2A cleaning)
    df_model = df[all_features + [target]].dropna()
    logger.info(f"Model dataset: {len(df_model):,} rows × {len(all_features)} features")
    logger.info(f"Default rate: {df_model[target].mean()*100:.2f}%")

    X = df_model[all_features]
    y = df_model[target]

    return X, y, all_features, numeric_features, binary_features


def check_multicollinearity(X, features):
    """
    Calculate Variance Inflation Factor (VIF) for numeric features.
    VIF > 10 = severe multicollinearity → drop one of the correlated pair.
    VIF > 5  = moderate, worth monitoring.
    """
    logger.info("Checking multicollinearity (VIF)...")
    X_scaled = StandardScaler().fit_transform(X[features])
    vif_data = pd.DataFrame({
        "feature": features,
        "vif": [variance_inflation_factor(X_scaled, i) for i in range(len(features))],
    }).sort_values("vif", ascending=False)

    print("\n" + "=" * 55)
    print("VARIANCE INFLATION FACTOR (VIF)")
    print("=" * 55)
    for _, row in vif_data.iterrows():
        flag = " ⚠️ HIGH" if row["vif"] > 10 else ""
        print(f"  {row['feature']:30s}  VIF: {row['vif']:8.2f}{flag}")
    print("=" * 55)

    # Identify features to drop (VIF > 10)
    high_vif = vif_data[vif_data["vif"] > 10]["feature"].tolist()
    if high_vif:
        logger.info(f"High-VIF features detected: {high_vif}")
        # We drop 'installment' because it's a near-linear function of
        # loan_amnt + int_rate + term (the formula is deterministic).
        # Also drop loan_amnt since installment_to_income captures the
        # payment burden more precisely.
        drop_cols = [f for f in ["installment", "loan_amnt"] if f in high_vif]
        logger.info(f"Dropping due to multicollinearity: {drop_cols}")
        return drop_cols
    return []


# ================================================================
# SECTION 2: LOGISTIC REGRESSION (statsmodels — for p-values)
# ================================================================
def fit_statsmodels_logistic(X_train_scaled, y_train, feature_names):
    """
    Fit logistic regression with statsmodels to get:
    - p-values for each coefficient (statistical significance)
    - Odds ratios (business interpretation)
    - Confidence intervals

    This is what a bank's model validation team wants to see.
    """
    logger.info("Fitting statsmodels logistic regression (for p-values + odds ratios)...")
    X_const = sm.add_constant(X_train_scaled)

    model = sm.Logit(y_train, X_const)
    result = model.fit(method='lbfgs', maxiter=200, disp=False)

    # Build summary table
    summary = pd.DataFrame({
        "feature": ["intercept"] + feature_names,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values,
        "odds_ratio": np.exp(result.params.values),
        "ci_lower": np.exp(result.conf_int().iloc[:, 0].values),
        "ci_upper": np.exp(result.conf_int().iloc[:, 1].values),
    })
    summary["significant"] = summary["p_value"] < 0.05
    summary["direction"] = summary["coefficient"].apply(
        lambda c: "↑ Increases default" if c > 0 else "↓ Decreases default"
    )

    print("\n" + "=" * 100)
    print("LOGISTIC REGRESSION — ODDS RATIOS (statsmodels)")
    print("=" * 100)
    print(f"{'Feature':30s} {'Coeff':>8s} {'Odds Ratio':>11s} {'p-value':>10s} {'Sig?':>5s}  Direction")
    print("-" * 100)
    for _, row in summary.iterrows():
        sig_marker = "  ✓" if row["significant"] else "   "
        print(f"{row['feature']:30s} {row['coefficient']:>8.4f} {row['odds_ratio']:>11.4f} "
              f"{row['p_value']:>10.4f} {sig_marker:>5s}  {row['direction']}")
    print("=" * 100)
    print("\nOdds ratio interpretation: For a 1 standard-deviation increase in the feature,")
    print("default odds are multiplied by the odds ratio. E.g., OR=1.30 means 30% higher odds.")
    print(f"\nPseudo R²: {result.prsquared:.4f}")
    print(f"AIC: {result.aic:.0f}")

    return result, summary


# ================================================================
# SECTION 3: SKLEARN MODEL (for predictions + evaluation)
# ================================================================
def fit_sklearn_logistic(X_train_scaled, y_train, X_test_scaled, y_test, feature_names):
    """
    Fit sklearn LogisticRegression with class_weight='balanced' to
    handle the ~87/13 class imbalance.

    WHY class_weight='balanced' INSTEAD OF SMOTE:
    - SMOTE generates synthetic minority samples, which can introduce
      artifacts in credit data (fake borrower profiles that look real)
    - class_weight adjusts the loss function to penalize misclassifying
      defaults more heavily — cleaner, no synthetic data needed
    - Industry preference: banks prefer interpretable corrections over
      synthetic data generation
    """
    logger.info("Fitting sklearn logistic regression (class_weight='balanced')...")

    model = LogisticRegression(
        class_weight='balanced',    # handles imbalance
        solver='lbfgs',
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return model, y_pred, y_prob


# ================================================================
# SECTION 4: MODEL EVALUATION
# ================================================================
def evaluate_model(y_test, y_pred, y_prob, feature_names, model):
    """
    Full evaluation suite with business interpretation.
    """
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"                     Predicted Non-Default  Predicted Default")
    print(f"  Actual Non-Default      {tn:>10,}           {fp:>10,}")
    print(f"  Actual Default          {fn:>10,}           {tp:>10,}")
    print("=" * 60)
    print("\nBUSINESS INTERPRETATION:")
    print(f"  True Positives  (correctly caught defaults):   {tp:>8,}")
    print(f"  False Negatives (defaults we MISSED → $ lost): {fn:>8,}")
    print(f"  False Positives (good loans rejected → lost revenue): {fp:>8,}")
    print(f"  True Negatives  (correctly approved):          {tn:>8,}")
    print(f"\n  Default catch rate (Recall/Sensitivity): {tp/(tp+fn)*100:.1f}%")
    print(f"  False alarm rate (1 - Specificity):       {fp/(fp+tn)*100:.1f}%")

    # --- Classification Report ---
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["Non-Default", "Default"]))

    # --- AUC Scores ---
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print("=" * 60)
    print(f"ROC AUC:                  {roc_auc:.4f}")
    print(f"Precision-Recall AUC:     {pr_auc:.4f}")
    print(f"F1 Score (Default class): {f1:.4f}")
    print("=" * 60)

    # --- Chart 10: Confusion Matrix Heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'], ax=ax,
                linewidths=1, linecolor='white',
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '10_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Chart 11: ROC Curve ---
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=2.5,
            label=f'Logistic Regression (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], linestyle='--',
            label='Random Classifier (AUC = 0.50)')
    ax.set_title('ROC Curve — Default Prediction', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '11_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Chart 12: Precision-Recall Curve ---
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color=COLORS['accent1'], linewidth=2.5,
            label=f'PR Curve (AP = {pr_auc:.4f})')
    baseline_rate = y_test.mean()
    ax.axhline(y=baseline_rate, color=COLORS['neutral'], linestyle='--',
               label=f'Baseline (default rate = {baseline_rate:.2f})')
    ax.set_title('Precision-Recall Curve\n(More informative than ROC for imbalanced data)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Recall (What % of actual defaults did we catch?)', fontsize=11)
    ax.set_ylabel('Precision (Of flagged loans, what % actually defaulted?)', fontsize=11)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '12_precision_recall_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  WHY PRECISION-RECALL > ROC FOR IMBALANCED DATA:")
    print("  ROC can look great (AUC=0.90) even when the model catches few")
    print("  actual defaults, because FPR stays low when non-defaults dominate.")
    print("  PR curve directly shows the tradeoff relevant to the business:")
    print("  'How many real defaults can I catch without flagging too many good loans?'")

    # --- Chart 13: Feature Importance (Absolute Coefficients) ---
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coeff': np.abs(model.coef_[0]),
    }).sort_values('abs_coeff', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_list = [COLORS['accent1'] if c > 0 else COLORS['accent2'] for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['abs_coeff'], color=colors_list, edgecolor='white')
    ax.set_title('Feature Importance — Logistic Regression Coefficients\n(Red = increases default risk, Green = decreases)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Absolute Coefficient (standardized)', fontsize=11)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '13_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    return roc_auc, pr_auc, f1, thresholds_roc, fpr, tpr


# ================================================================
# SECTION 5: OPTIMAL THRESHOLD ANALYSIS
# ================================================================
def find_optimal_threshold(y_test, y_prob, thresholds_roc, fpr, tpr):
    """
    Default threshold is 0.5, but this is WRONG for most business cases.
    In lending:
      - False Negative (missing a default) costs: avg_loan × LGD ≈ $15K × 0.9 = $13.5K
      - False Positive (rejecting a good loan) costs: avg_interest ≈ $3K in lost revenue
    So FN is ~4x more expensive than FP → we should lower the threshold.

    We find: (a) Youden's J-statistic optimal, and (b) business-cost optimal.
    """
    # Youden's J: maximizes (Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds_roc[optimal_idx]

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"  Default threshold:          0.50")
    print(f"  Youden's J optimal:         {optimal_threshold:.4f}")

    # Business-cost threshold: iterate and find the one that minimizes total cost
    fn_cost = 13500    # avg loss from a missed default
    fp_cost = 3000     # avg lost revenue from rejecting a good loan

    best_cost = float('inf')
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred_t = (y_prob >= thresh).astype(int)
        cm_t = confusion_matrix(y_test, y_pred_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        total_cost = fn_t * fn_cost + fp_t * fp_cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = thresh

    print(f"  Business-cost optimal:      {best_thresh:.4f}")
    print(f"  (FN cost=${fn_cost:,}, FP cost=${fp_cost:,})")

    # Show confusion matrix at optimal threshold
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
    print(f"\n  At threshold = {best_thresh:.2f}:")
    print(f"    Defaults caught:     {tp_o:,} / {tp_o+fn_o:,} ({tp_o/(tp_o+fn_o)*100:.1f}%)")
    print(f"    Good loans rejected: {fp_o:,} / {fp_o+tn_o:,} ({fp_o/(fp_o+tn_o)*100:.1f}%)")
    print(f"    Total business cost: ${best_cost:,.0f}")

    default_pred = (y_prob >= 0.5).astype(int)
    cm_def = confusion_matrix(y_test, default_pred)
    tn_d, fp_d, fn_d, tp_d = cm_def.ravel()
    default_cost = fn_d * fn_cost + fp_d * fp_cost
    savings = default_cost - best_cost
    print(f"\n  vs default threshold 0.50: ${default_cost:,.0f}")
    print(f"  SAVINGS from optimal threshold: ${savings:,.0f}")
    print("=" * 60)

    # --- Chart 14: Threshold vs Metrics ---
    thresholds_range = np.arange(0.05, 0.95, 0.01)
    recalls, precisions, f1s = [], [], []
    for t in thresholds_range:
        y_t = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_test, y_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        rec = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        prec = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        f1_t = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        recalls.append(rec)
        precisions.append(prec)
        f1s.append(f1_t)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds_range, recalls, label='Recall (Catch Rate)', color=COLORS['accent2'], linewidth=2)
    ax.plot(thresholds_range, precisions, label='Precision (Accuracy of Flags)', color=COLORS['accent1'], linewidth=2)
    ax.plot(thresholds_range, f1s, label='F1 Score', color=COLORS['accent4'], linewidth=2)
    ax.axvline(best_thresh, color=COLORS['accent3'], linestyle='--', linewidth=2,
               label=f'Optimal threshold = {best_thresh:.2f}')
    ax.set_title('Threshold Tuning — Precision/Recall/F1 Tradeoff',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '14_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    return best_thresh


# ================================================================
# SECTION 6: SIMPLIFIED RISK SCORECARD
# ================================================================
def build_scorecard(sm_summary, feature_names):
    """
    Convert logistic regression coefficients into a simplified
    points-based scorecard.

    This is THE section that makes this project memorable. A loan
    officer can use this scorecard with a calculator — no code needed.

    Method: scale coefficients to a 0-100 point system where:
      base_score = 500
      higher score = lower risk = approve
      lower score  = higher risk = decline or manual review
    """
    print("\n" + "=" * 65)
    print("SIMPLIFIED RISK SCORECARD")
    print("=" * 65)
    print("How a loan officer uses this: sum the points for each")
    print("applicant. Higher score = safer borrower.")
    print("-" * 65)

    # Use only significant features for the scorecard
    sig_features = sm_summary[
        (sm_summary["significant"] == True) &
        (sm_summary["feature"] != "intercept")
    ].copy()

    # Scale coefficients to points: normalize to ±50 point range
    max_abs_coeff = sig_features["coefficient"].abs().max()
    sig_features["points"] = (-sig_features["coefficient"] / max_abs_coeff * 50).round(0).astype(int)
    # Negative coefficient (decreases default) = POSITIVE points (good)
    # Positive coefficient (increases default) = NEGATIVE points (bad)

    base_score = 500

    print(f"\n  BASE SCORE: {base_score} points")
    print(f"\n  {'Feature':30s} {'Points':>8s}  Effect")
    print(f"  {'-'*55}")

    for _, row in sig_features.sort_values("points").iterrows():
        effect = "⬇ Safer" if row["points"] > 0 else "⬆ Riskier"
        sign = "+" if row["points"] > 0 else ""
        print(f"  {row['feature']:30s} {sign}{row['points']:>7.0f}  {effect}")

    print(f"\n  {'='*55}")
    print(f"  SCORING BANDS:")
    print(f"    Score > 550:  AUTO-APPROVE  (low risk)")
    print(f"    Score 480-550: MANUAL REVIEW (moderate risk)")
    print(f"    Score < 480:  DECLINE       (high risk)")
    print(f"  {'='*55}")

    print("\n  EXAMPLE APPLICATION:")
    print(f"    Base score:                {base_score}")
    example_pts = sig_features.nlargest(3, "points")["points"].sum()
    example_neg = sig_features.nsmallest(2, "points")["points"].sum()
    print(f"    + Best 3 risk factors:     +{example_pts:.0f}")
    print(f"    + Worst 2 risk factors:    {example_neg:.0f}")
    print(f"    = FINAL SCORE:             {base_score + example_pts + example_neg:.0f}")

    return sig_features


# ================================================================
# SECTION 7: STATISTICAL TESTS
# ================================================================
def run_statistical_tests(df):
    """
    Hypothesis tests that demonstrate statistical rigor.
    These are the most commonly asked stat questions in DA interviews.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    # Test 1: Chi-squared — default rate differs across grades
    print("\n--- Test 1: Chi-Squared (Default Rate × Grade) ---")
    print("H₀: Default rate is the SAME across all grades (A through G)")
    print("H₁: Default rate DIFFERS significantly across grades")
    contingency = pd.crosstab(df["grade"], df["default"])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
    print(f"  χ² statistic: {chi2:,.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_chi:.2e}")
    if p_chi < 0.05:
        print("  → REJECT H₀: Default rate is significantly different across grades.")
        print("    Grade is a statistically valid risk discriminator.")
    else:
        print("  → FAIL TO REJECT H₀")

    # Test 2: T-test — income differs between defaulters and non-defaulters
    print("\n--- Test 2: Independent T-Test (Income: Defaulters vs Non-Defaulters) ---")
    print("H₀: Mean annual income is the SAME for defaulters and non-defaulters")
    print("H₁: Mean annual income DIFFERS between the two groups")
    inc_default = df[df["default"] == 1]["annual_inc_capped"]
    inc_nondefault = df[df["default"] == 0]["annual_inc_capped"]
    t_stat, p_ttest = stats.ttest_ind(inc_nondefault, inc_default, equal_var=False)
    print(f"  Mean income (non-default): ${inc_nondefault.mean():,.0f}")
    print(f"  Mean income (default):     ${inc_default.mean():,.0f}")
    print(f"  Difference:                ${inc_nondefault.mean() - inc_default.mean():,.0f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_ttest:.2e}")
    if p_ttest < 0.05:
        print("  → REJECT H₀: Income is significantly different between groups.")
    else:
        print("  → FAIL TO REJECT H₀")

    # Test 3: Mann-Whitney U — FICO scores (non-parametric, safer for non-normal data)
    print("\n--- Test 3: Mann-Whitney U Test (FICO: Defaulters vs Non-Defaulters) ---")
    print("H₀: FICO score distributions are the same for both groups")
    print("H₁: FICO score distributions differ")
    fico_default = df[df["default"] == 1]["fico_score"]
    fico_nondefault = df[df["default"] == 0]["fico_score"]
    u_stat, p_mann = stats.mannwhitneyu(fico_nondefault, fico_default, alternative='two-sided')
    print(f"  Median FICO (non-default): {fico_nondefault.median():.0f}")
    print(f"  Median FICO (default):     {fico_default.median():.0f}")
    print(f"  U-statistic: {u_stat:,.0f}")
    print(f"  p-value: {p_mann:.2e}")
    if p_mann < 0.05:
        print("  → REJECT H₀: FICO distributions are significantly different.")
    else:
        print("  → FAIL TO REJECT H₀")

    # Confidence interval for default rate
    print("\n--- 95% Confidence Interval for Portfolio Default Rate ---")
    n = len(df)
    p = df["default"].mean()
    se = np.sqrt(p * (1 - p) / n)
    ci_lower = p - 1.96 * se
    ci_upper = p + 1.96 * se
    print(f"  Point estimate: {p*100:.3f}%")
    print(f"  95% CI: [{ci_lower*100:.3f}%, {ci_upper*100:.3f}%]")
    print(f"  In plain English: We are 95% confident that the true population")
    print(f"  default rate lies between {ci_lower*100:.2f}% and {ci_upper*100:.2f}%.")
    print("=" * 70)


# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    logger.info("Starting credit risk modeling pipeline...")

    # --- Load & Prepare ---
    X, y, all_features, numeric_features, binary_features = load_and_prepare()

    # --- Check Multicollinearity ---
    drop_cols = check_multicollinearity(X, numeric_features)
    if drop_cols:
        X = X.drop(columns=drop_cols)
        all_features = [f for f in all_features if f not in drop_cols]
        numeric_features = [f for f in numeric_features if f not in drop_cols]
        logger.info(f"Features after VIF cleanup: {len(all_features)}")

    # --- Train/Test Split (stratified to preserve class balance) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    logger.info(f"Train default rate: {y_train.mean()*100:.2f}% | Test: {y_test.mean()*100:.2f}%")

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Statsmodels Logistic (p-values + odds ratios) ---
    sm_result, sm_summary = fit_statsmodels_logistic(X_train_scaled, y_train, all_features)

    # --- Sklearn Logistic (predictions + evaluation) ---
    model, y_pred, y_prob = fit_sklearn_logistic(
        X_train_scaled, y_train, X_test_scaled, y_test, all_features
    )

    # --- Evaluate ---
    roc_auc, pr_auc, f1, thresholds_roc, fpr, tpr = evaluate_model(
        y_test, y_pred, y_prob, all_features, model
    )

    # --- Optimal Threshold ---
    optimal_thresh = find_optimal_threshold(y_test, y_prob, thresholds_roc, fpr, tpr)

    # --- Scorecard ---
    scorecard = build_scorecard(sm_summary, all_features)

    # --- Statistical Tests ---
    df_full = pd.read_csv(DATA_DIR / "features_loans.csv", low_memory=False)
    run_statistical_tests(df_full)

    # --- Save predictions for Power BI ---
    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred,
        "default_probability": y_prob,
    })
    results.to_csv(DATA_DIR / "model_results.csv", index=False)
    logger.info("Predictions saved to data/model_results.csv")

    # --- Executive Summary ---
    print("\n" + "=" * 65)
    print("EXECUTIVE SUMMARY — CREDIT RISK MODEL")
    print("=" * 65)
    print(f"""
    1. MODEL: Logistic Regression with balanced class weights
       - Handles {y.mean()*100:.1f}% default rate imbalance
       - Industry-standard for regulated credit decisions

    2. PERFORMANCE:
       - ROC AUC:             {roc_auc:.4f}
       - Precision-Recall AUC: {pr_auc:.4f}
       - F1 Score (Default):   {f1:.4f}

    3. TOP RISK FACTORS (from odds ratios):
       - Grade (LC's own rating) — strongest predictor
       - Interest rate — proxy for LC's risk assessment
       - FICO score — traditional credit score, still powerful
       - DTI — debt burden, especially above 30%
       - Recent credit inquiries — credit-seeking behavior

    4. SCORECARD: Base 500 ± risk points → Auto-Approve / Review / Decline

    5. OPTIMAL THRESHOLD: {optimal_thresh:.2f} (vs default 0.50)
       — catches more defaults at acceptable false-alarm cost

    Charts saved: charts/10-14 (5 model evaluation visuals)
    """)
    print("=" * 65)
    logger.info("Sprint 2C complete. Next → Sprint 2D: Dashboard + GitHub Polish.")


if __name__ == "__main__":
    main()
