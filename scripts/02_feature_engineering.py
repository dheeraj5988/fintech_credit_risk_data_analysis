#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lending Club Credit Risk — Feature Engineering
================================================================
Sprint 2B (Part 2): Create model-ready features with business logic.

Author: Dheeraj Sharma

Every feature here is INTERVIEW-DEFENSIBLE — each has a clear
1-line risk rationale (see FEATURE_RATIONALE at the bottom, which
also prints when you run the script).

Run:
    python scripts/02_feature_engineering.py

Input:  data/cleaned_loans.csv   (from Sprint 2A)
Output: data/features_loans.csv  (for Sprint 2C modeling)
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "cleaned_loans.csv"
OUTPUT_PATH = DATA_DIR / "features_loans.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk features. Each block is one feature + its rationale."""
    df = df.copy()

    # 1. income_to_loan_ratio — repayment capacity relative to obligation.
    #    Higher = borrower earns many multiples of the loan = safer.
    df["income_to_loan_ratio"] = df["annual_inc_capped"] / df["loan_amnt"].replace(0, np.nan)
    df["income_to_loan_ratio"] = df["income_to_loan_ratio"].clip(upper=df["income_to_loan_ratio"].quantile(0.99))

    # 2. installment_to_monthly_income — payment shock measure.
    #    What % of monthly income goes to THIS loan's payment alone?
    monthly_income = df["annual_inc_capped"] / 12
    df["installment_to_income"] = (df["installment"] / monthly_income.replace(0, np.nan)).clip(upper=1)

    # 3. high_dti_flag — binary underwriting cutoff signal.
    #    SQL Q5 showed default acceleration above DTI 30; flag captures the knee.
    df["high_dti_flag"] = (df["dti"] > 30).astype(int)

    # 4. revol_util_bucket — credit stress categories.
    #    >80% utilization = borrower is maxed out = classic distress signal.
    df["revol_util_bucket"] = pd.cut(
        df["revol_util"],
        bins=[-0.01, 30, 60, 80, 200],
        labels=["low_0_30", "moderate_30_60", "high_60_80", "maxed_80plus"],
    )

    # 5. emp_stability — employment length as ordinal categories.
    #    'unknown' kept separate: SQL Q7 showed non-reporters default MOST.
    def emp_bucket(row):
        if row["emp_length_missing"] == 1:
            return "unknown"
        y = row["emp_length_years"]
        if y < 1:
            return "under_1yr"
        if y <= 3:
            return "1_3yrs"
        if y <= 9:
            return "4_9yrs"
        return "10plus"
    df["emp_stability"] = df.apply(emp_bucket, axis=1)

    # 6. grade_numeric — ordinal encoding of LC's grade.
    #    A=1 ... G=7. Preserves the monotonic risk ordering shown in SQL Q2.
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_numeric"] = df["grade"].map(grade_map)

    # 7. fico_band — standard industry credit score bands.
    df["fico_band"] = pd.cut(
        df["fico_score"],
        bins=[0, 670, 740, 900],
        labels=["fair_under670", "good_670_739", "verygood_740plus"],
    )

    # 8. debt_consolidation_flag — purpose signal.
    #    Consolidators are refinancing existing debt (moderate risk);
    #    contrast with small_business (high risk) captured separately.
    df["debt_consolidation_flag"] = (df["purpose"] == "debt_consolidation").astype(int)
    df["small_business_flag"] = (df["purpose"] == "small_business").astype(int)

    # 9. recent_inquiry_flag — credit-seeking behavior.
    #    2+ hard inquiries in 6 months = borrower actively hunting credit
    #    = liquidity stress signal (validated in SQL Q6 profile comparison).
    df["recent_inquiry_flag"] = (df["inq_last_6mths"] >= 2).astype(int)

    # 10. derogatory_flag — any public record or bankruptcy.
    #     Combines pub_rec + bankruptcies into one clean binary signal.
    df["derogatory_flag"] = ((df["pub_rec"] > 0) | (df["pub_rec_bankruptcies"] > 0)).astype(int)

    # 11. long_term_flag — 60-month loans carry more uncertainty than 36.
    df["long_term_flag"] = (df["term"] == 60).astype(int)

    # 12. Interaction: grade_x_dti — does high DTI hurt MORE in risky grades?
    #     Captures compounding risk (used cautiously; explained in modeling).
    df["grade_x_dti"] = df["grade_numeric"] * df["dti"]

    return df


FEATURE_RATIONALE = """
FEATURE RATIONALE (memorize these one-liners for interviews):
──────────────────────────────────────────────────────────────
income_to_loan_ratio    → Repayment capacity: income as multiple of loan size
installment_to_income   → Payment shock: % of monthly income eaten by this payment
high_dti_flag           → Underwriting cutoff at the default-acceleration knee (DTI>30)
revol_util_bucket       → Maxed-out credit cards = classic financial distress signal
emp_stability           → Employment tenure; 'unknown' kept separate (non-reporting = risk)
grade_numeric           → LC's own risk model, ordinal-encoded to preserve monotonic ordering
fico_band               → Industry-standard credit score bands
debt_consolidation_flag → Refinancing behavior differs from new-spending borrowers
small_business_flag     → Highest-default purpose category (business income volatility)
recent_inquiry_flag     → 2+ recent inquiries = active credit-seeking = liquidity stress
derogatory_flag         → Any bankruptcy/public record — the strongest single red flag
long_term_flag          → 60-month terms = more time for borrower circumstances to worsen
grade_x_dti             → Interaction: high DTI is more dangerous in already-risky grades
"""


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH} — run scripts/01_clean_and_eda.py first")

    logger.info("Loading cleaned data...")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    df = engineer_features(df)

    new_features = [
        "income_to_loan_ratio", "installment_to_income", "high_dti_flag",
        "revol_util_bucket", "emp_stability", "grade_numeric", "fico_band",
        "debt_consolidation_flag", "small_business_flag", "recent_inquiry_flag",
        "derogatory_flag", "long_term_flag", "grade_x_dti",
    ]
    logger.info(f"Created {len(new_features)} features")

    # Quick validation: default rate by each new binary flag
    print("\n" + "=" * 60)
    print("FEATURE VALIDATION — default rate by flag")
    print("=" * 60)
    for flag in ["high_dti_flag", "recent_inquiry_flag", "derogatory_flag",
                 "small_business_flag", "long_term_flag"]:
        rates = df.groupby(flag)["default"].mean() * 100
        print(f"{flag:25s}  flag=0: {rates.get(0, float('nan')):.2f}%   flag=1: {rates.get(1, float('nan')):.2f}%")
    print("=" * 60)
    print(FEATURE_RATIONALE)

    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Feature dataset saved: {OUTPUT_PATH}")
    logger.info("Sprint 2B complete. Next: Sprint 2C — Modeling + Statistics.")


if __name__ == "__main__":
    main()
