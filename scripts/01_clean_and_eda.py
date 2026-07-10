#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lending Club Credit Risk — Data Cleaning & Risk EDA Pipeline
================================================================
Sprint 2A: Load, clean, engineer target variable, and produce
risk-focused exploratory analysis.

Author: Dheeraj Sharma
Database: credit_risk_analytics (MySQL)

Run:
    python scripts/01_clean_and_eda.py

Requires:
    - accepted_2007_to_2018Q4.csv.gz in data/raw/
    - MySQL running with `credit_risk_analytics` database created
    - .env file with DB credentials
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "credit_risk_analytics")

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
CHARTS_DIR = Path(__file__).parent.parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# If your machine is slow / low on RAM, lower this (e.g. 0.3 = use 30% of rows).
# 1.0 = use all ~2.26M rows. Sampling is done AFTER cleaning, stratified by
# target, so class balance is preserved.
SAMPLE_FRAC = 1.0
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Chart Style (consistent with Project 1) ---
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


def get_engine():
    conn_str = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


# ----------------------------------------------------------------------
# COLUMNS TO KEEP (avoids loading 151 columns of mostly-leakage/junk)
# ----------------------------------------------------------------------
KEEP_COLUMNS = [
    # Loan characteristics (known at application time)
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "title",
    "zip_code", "addr_state", "dti",
    # Credit history (known at application time)
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "initial_list_status", "application_type", "mort_acc",
    "pub_rec_bankruptcies", "acc_now_delinq",
]

# Columns explicitly excluded — see docs/01_DATASET_AND_RISK_CONTEXT.md
# for full leakage explanation. Not referenced directly since we use an
# allow-list (KEEP_COLUMNS) rather than a deny-list, which is safer.


# ----------------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------------
def load_raw_data() -> pd.DataFrame:
    """Load only the columns we need from the large Lending Club CSV."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {RAW_DATA_PATH}\n"
            f"Download from Kaggle and place in data/raw/. "
            f"See docs/01_DATASET_AND_RISK_CONTEXT.md"
        )
    logger.info("Loading Lending Club data (this may take 1-3 minutes)...")
    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=KEEP_COLUMNS,
        compression="gzip",
        low_memory=False,
    )
    logger.info(f"Loaded raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------------
# TARGET ENGINEERING
# ----------------------------------------------------------------------
def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse loan_status into a binary target.

    Default = 1: Charged Off, Default, Late (31-120 days)
    Non-Default = 0: Fully Paid, Current
    Excluded: In Grace Period, Late (16-30 days), Issued
              (outcome not yet resolved — can't label these)
    """
    df = df.copy()

    default_statuses = ["Charged Off", "Default", "Late (31-120 days)"]
    nondefault_statuses = ["Fully Paid", "Current"]

    before = len(df)
    df = df[df["loan_status"].isin(default_statuses + nondefault_statuses)].copy()
    excluded = before - len(df)
    logger.info(f"Excluded {excluded:,} rows with unresolved loan_status (Grace Period, etc.)")

    df["default"] = df["loan_status"].isin(default_statuses).astype(int)

    default_rate = df["default"].mean() * 100
    logger.info(f"Target engineered — overall default rate: {default_rate:.2f}%")
    logger.info(f"Class balance — Default: {df['default'].sum():,} | Non-Default: {(df['default']==0).sum():,}")

    return df


# ----------------------------------------------------------------------
# CLEANING
# ----------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Lending Club dataset with risk-context justified decisions.
    """
    df = df.copy()

    # --- Term: "36 months" -> 36 (int) ---
    df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)

    # --- emp_length: "10+ years" -> 10, "< 1 year" -> 0, NaN kept as NaN ---
    def parse_emp_length(val):
        if pd.isna(val):
            return np.nan
        val = str(val)
        if "10+" in val:
            return 10
        if "< 1" in val:
            return 0
        digits = "".join(c for c in val if c.isdigit())
        return float(digits) if digits else np.nan

    df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
    # Missing emp_length is itself a signal (often unemployed/self-employed
    # borrowers skip this field) — impute with -1 flag rather than mean,
    # so the model/analysis can treat "unknown employment" as its own category
    df["emp_length_missing"] = df["emp_length_years"].isna().astype(int)
    df["emp_length_years"] = df["emp_length_years"].fillna(-1)

    # --- dti: a small number of extreme/negative values exist (data errors) ---
    # DTI over 100% or negative is not economically meaningful — cap, don't drop,
    # to preserve row count for other analyses
    df["dti"] = df["dti"].clip(lower=0, upper=100)
    df["dti"] = df["dti"].fillna(df["dti"].median())

    # --- annual_inc: cap extreme outliers (a few rows report >$5M income,
    # almost certainly data entry errors) using 99.5th percentile ---
    income_cap = df["annual_inc"].quantile(0.995)
    df["annual_inc_capped"] = df["annual_inc"].clip(upper=income_cap)
    df["annual_inc_capped"] = df["annual_inc_capped"].fillna(df["annual_inc_capped"].median())

    # --- revol_util: comes as string with '%', some nulls ---
    if df["revol_util"].dtype == object:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "", regex=False)
        df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")
    df["revol_util"] = df["revol_util"].fillna(df["revol_util"].median())

    # --- mort_acc, pub_rec_bankruptcies: null usually means 0 (no mortgages /
    # no bankruptcies on file), which is the sensible imputation here ---
    df["mort_acc"] = df["mort_acc"].fillna(0)
    df["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)

    # --- fico: use midpoint of range for a single feature ---
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # --- issue_d: parse to datetime for vintage analysis ---
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["issue_year"] = df["issue_date"].dt.year

    # --- Drop rows still missing core underwriting fields we can't safely impute ---
    critical_cols = ["loan_amnt", "int_rate", "grade", "dti", "annual_inc_capped", "fico_score"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    logger.info(f"Dropped {before - len(df):,} rows missing critical underwriting fields")

    return df


# ----------------------------------------------------------------------
# RISK EDA — 8+ VISUALIZATIONS
# ----------------------------------------------------------------------
def plot_default_rate_by_grade(df):
    """Chart 1: Default rate by grade — THE core risk visual."""
    grade_stats = df.groupby("grade")["default"].agg(["mean", "count"]).reset_index()
    grade_stats["default_rate"] = grade_stats["mean"] * 100
    grade_stats = grade_stats.sort_values("grade")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(grade_stats["grade"], grade_stats["default_rate"],
                   color=COLORS['primary'], edgecolor='white')
    # Color-code by risk severity
    for bar, rate in zip(bars, grade_stats["default_rate"]):
        if rate > 25:
            bar.set_color(COLORS['accent1'])
        elif rate > 15:
            bar.set_color(COLORS['accent3'])
        else:
            bar.set_color(COLORS['accent2'])

    for bar, val, n in zip(bars, grade_stats["default_rate"], grade_stats["count"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.1f}%\n(n={n:,})', ha='center', fontsize=9)

    ax.set_title('Default Rate by Loan Grade', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Grade (A = Safest, G = Riskiest)', fontsize=11)
    ax.set_ylabel('Default Rate (%)', fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '01_default_rate_by_grade.png', dpi=150, bbox_inches='tight')
    plt.show()
    return grade_stats


def plot_default_rate_by_purpose(df):
    """Chart 2: Default rate by loan purpose."""
    purpose_stats = (
        df.groupby("purpose")["default"]
        .agg(["mean", "count"])
        .reset_index()
    )
    purpose_stats = purpose_stats[purpose_stats["count"] >= 1000]
    purpose_stats["default_rate"] = purpose_stats["mean"] * 100
    purpose_stats = purpose_stats.sort_values("default_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(purpose_stats["purpose"], purpose_stats["default_rate"],
                    color=COLORS['primary'], edgecolor='white')
    for bar in bars[-3:]:
        bar.set_color(COLORS['accent1'])
    ax.set_title('Default Rate by Loan Purpose', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Default Rate (%)', fontsize=11)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '02_default_rate_by_purpose.png', dpi=150, bbox_inches='tight')
    plt.show()
    return purpose_stats


def plot_default_by_home_emp(df):
    """Chart 3: Default rate by home ownership and employment length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    home_stats = df.groupby("home_ownership")["default"].mean() * 100
    home_stats = home_stats.sort_values(ascending=False)
    axes[0].bar(home_stats.index, home_stats.values, color=COLORS['primary'], edgecolor='white')
    axes[0].set_title('Default Rate by Home Ownership', fontweight='bold')
    axes[0].set_ylabel('Default Rate (%)')

    emp_stats = df[df["emp_length_missing"] == 0].groupby("emp_length_years")["default"].mean() * 100
    axes[1].plot(emp_stats.index, emp_stats.values, marker='o', color=COLORS['primary'], linewidth=2)
    axes[1].set_title('Default Rate by Employment Length', fontweight='bold')
    axes[1].set_xlabel('Years Employed')
    axes[1].set_ylabel('Default Rate (%)')

    for ax in axes:
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '03_default_by_home_emp.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_interest_rate_vs_default(df):
    """Chart 4: Interest rate distribution split by default outcome."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df["default"] == 0]["int_rate"], bins=50, alpha=0.6,
            color=COLORS['accent2'], label='Non-Default', density=True)
    ax.hist(df[df["default"] == 1]["int_rate"], bins=50, alpha=0.6,
            color=COLORS['accent1'], label='Default', density=True)
    ax.set_title('Interest Rate Distribution: Default vs Non-Default',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Interest Rate (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '04_interest_rate_vs_default.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_dti_distribution(df):
    """Chart 5: DTI distribution — defaulters vs non-defaulters (overlapping histogram)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df["default"] == 0]["dti"], bins=50, alpha=0.6,
            color=COLORS['accent2'], label='Non-Default', density=True)
    ax.hist(df[df["default"] == 1]["dti"], bins=50, alpha=0.6,
            color=COLORS['accent1'], label='Default', density=True)
    ax.set_title('DTI (Debt-to-Income) Distribution: Default vs Non-Default',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('DTI (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '05_dti_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_loan_amount_by_outcome(df):
    """Chart 6: Loan amount distribution by outcome."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="default", y="loan_amnt", ax=ax,
                palette=[COLORS['accent2'], COLORS['accent1']])
    ax.set_xticklabels(['Non-Default', 'Default'])
    ax.set_title('Loan Amount Distribution by Outcome', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('')
    ax.set_ylabel('Loan Amount ($)', fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '06_loan_amount_by_outcome.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df):
    """Chart 7: Correlation heatmap of numerical features with default."""
    num_cols = [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc_capped",
        "dti", "delinq_2yrs", "fico_score", "inq_last_6mths", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
        "pub_rec_bankruptcies", "default"
    ]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Heatmap — Numerical Features vs Default',
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    return corr


def plot_default_by_state(df):
    """Chart 8: Geographic default rate by state."""
    state_stats = df.groupby("addr_state")["default"].agg(["mean", "count"]).reset_index()
    state_stats = state_stats[state_stats["count"] >= 500]
    state_stats["default_rate"] = state_stats["mean"] * 100
    state_stats = state_stats.sort_values("default_rate", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(state_stats["addr_state"], state_stats["default_rate"],
                    color=COLORS['accent1'], edgecolor='white')
    ax.invert_yaxis()
    ax.set_title('Top 15 States by Default Rate (min 500 loans)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Default Rate (%)', fontsize=11)
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '08_default_by_state.png', dpi=150, bbox_inches='tight')
    plt.show()
    return state_stats


def plot_fico_vs_default(df):
    """Chart 9 (bonus): FICO score distribution by outcome."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df["default"] == 0]["fico_score"], bins=40, alpha=0.6,
            color=COLORS['accent2'], label='Non-Default', density=True)
    ax.hist(df[df["default"] == 1]["fico_score"], bins=40, alpha=0.6,
            color=COLORS['accent1'], label='Default', density=True)
    ax.set_title('FICO Score Distribution: Default vs Non-Default',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('FICO Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=11)
    sns.despine()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / '09_fico_vs_default.png', dpi=150, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# MYSQL LOAD
# ----------------------------------------------------------------------
def load_to_mysql(df: pd.DataFrame, engine):
    """Load cleaned data into MySQL as a single flat table for SQL analysis."""
    cols_for_sql = [
        "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
        "emp_length_years", "emp_length_missing", "home_ownership",
        "annual_inc_capped", "verification_status", "issue_date", "issue_year",
        "purpose", "addr_state", "dti", "delinq_2yrs", "fico_score",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "mort_acc", "pub_rec_bankruptcies", "default",
    ]
    df_sql = df[cols_for_sql].copy()
    df_sql.to_sql("loans", engine, if_exists="replace", index=False, chunksize=10000)
    logger.info(f"Loaded {len(df_sql):,} rows into MySQL table `loans`")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    logger.info("Starting Lending Club credit risk pipeline...")

    raw = load_raw_data()
    df = engineer_target(raw)
    df = clean_data(df)

    if SAMPLE_FRAC < 1.0:
        df = df.groupby("default", group_keys=False).apply(
            lambda x: x.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        )
        logger.info(f"Sampled down to {len(df):,} rows (SAMPLE_FRAC={SAMPLE_FRAC})")

    print(f"\n{'='*70}\nFINAL CLEANED DATASET: {df.shape[0]:,} rows × {df.shape[1]} columns\n{'='*70}\n")

    # --- Risk EDA ---
    logger.info("Generating risk visualizations...")
    grade_stats = plot_default_rate_by_grade(df)
    purpose_stats = plot_default_rate_by_purpose(df)
    plot_default_by_home_emp(df)
    plot_interest_rate_vs_default(df)
    plot_dti_distribution(df)
    plot_loan_amount_by_outcome(df)
    corr = plot_correlation_heatmap(df)
    state_stats = plot_default_by_state(df)
    plot_fico_vs_default(df)

    # --- Risk Findings Summary ---
    print(f"\n{'='*70}\nRISK FINDINGS SUMMARY\n{'='*70}")
    print(f"1. Overall default rate: {df['default'].mean()*100:.2f}%")
    print(f"2. Grade A default rate: {grade_stats[grade_stats['grade']=='A']['default_rate'].values[0]:.2f}%"
          f"  vs  Grade G: {grade_stats[grade_stats['grade']=='G']['default_rate'].values[0]:.2f}%"
          if 'G' in grade_stats['grade'].values else "")
    print(f"3. Riskiest purpose: {purpose_stats.iloc[-1]['purpose']} "
          f"({purpose_stats.iloc[-1]['default_rate']:.2f}%)")
    print(f"4. Strongest numeric correlation with default: "
          f"{corr['default'].drop('default').abs().idxmax()} "
          f"({corr['default'].drop('default').abs().max():.3f})")
    print(f"5. Riskiest state: {state_stats.iloc[0]['addr_state']} "
          f"({state_stats.iloc[0]['default_rate']:.2f}%)")
    print(f"{'='*70}\n")

    # --- Load to MySQL for Sprint 2B ---
    engine = get_engine()
    load_to_mysql(df, engine)

    # --- Save cleaned CSV for Sprint 2C modeling ---
    output_path = Path(__file__).parent.parent / "data" / "cleaned_loans.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")

    logger.info("Pipeline complete. Ready for Sprint 2B — SQL Risk Analytics.")


if __name__ == "__main__":
    main()
