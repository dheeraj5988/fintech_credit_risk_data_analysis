# Power BI Risk Dashboard Blueprint
# Project 2 — Fintech Loan Default & Credit Risk Analysis
# Author: Dheeraj Sharma
# ================================================================
#
# DATA SOURCES:
# 1. MySQL: credit_risk_analytics → table: loans
# 2. CSV: data/model_results.csv (predictions + probabilities)
# 3. CSV: data/features_loans.csv (engineered features)
#
# COLOR THEME (Risk-standard RAG):
#   High Risk / Danger:   #E74C3C (red)
#   Moderate / Warning:   #F39C12 (amber)
#   Low Risk / Safe:      #27AE60 (green)
#   Primary / Headers:    #1B4F72 (navy)
#   Background:           #FAFAFA
#   Card Background:      #FFFFFF
# ================================================================


# ================================================================
# DAX MEASURES — Create in a "Measures" table
# ================================================================

Total Loans = 
COUNTROWS(loans)

Total Loan Volume = 
SUM(loans[loan_amnt])

Overall Default Rate = 
DIVIDE(
    CALCULATE(COUNTROWS(loans), loans[default] = 1),
    COUNTROWS(loans),
    0
) * 100

Avg Interest Rate = 
AVERAGE(loans[int_rate])

Avg FICO Score = 
AVERAGE(loans[fico_score])

Avg DTI = 
AVERAGE(loans[dti])

Default Count = 
CALCULATE(COUNTROWS(loans), loans[default] = 1)

Non-Default Count = 
CALCULATE(COUNTROWS(loans), loans[default] = 0)

Expected Loss Per Loan = 
AVERAGEX(
    FILTER(loans, loans[default] = 1),
    loans[loan_amnt] * 0.90
)

Portfolio Expected Loss = 
[Overall Default Rate] / 100 * SUM(loans[loan_amnt]) * 0.90

Risk Adjusted Return = 
AVERAGE(loans[int_rate]) - ([Overall Default Rate] / 100 * 90)

-- Date table (for vintage analysis)
Date Table = 
CALENDAR(DATE(2007,1,1), DATE(2018,12,31))
-- Add columns: Year, Quarter, YearQuarter


# ================================================================
# PAGE 1: PORTFOLIO OVERVIEW
# ================================================================
# 
# ┌──────────────────────────────────────────────────────────────┐
# │  CREDIT RISK DASHBOARD — Portfolio Overview   [Year Slicer]  │
# ├──────────┬──────────┬──────────┬──────────┬─────────────────┤
# │  Total   │ Total $  │ Default  │   Avg    │  Expected Loss  │
# │  Loans   │ Volume   │  Rate %  │  FICO    │  Per Loan ($)   │
# ├──────────┴──────────┴──────────┴──────────┴─────────────────┤
# │                                                              │
# │     Default Rate Trend by Issue Year (Line Chart)            │
# │     X: issue_year  Y: [Overall Default Rate]                 │
# │     Color: #1B4F72   Add trendline                           │
# │     [Full width — 40% height]                                │
# │                                                              │
# ├────────────────────────────┬─────────────────────────────────┤
# │  Portfolio by Grade        │  Loan Purpose Breakdown          │
# │  (Stacked Bar)             │  (Donut Chart)                   │
# │  X: grade                  │  Values: Count by purpose        │
# │  Y: loan count             │  Top 5 purposes only             │
# │  Stack: default / non-def  │  Colors: palette sequence        │
# │  Colors: Red / Green       │                                  │
# │  [50% width]               │  [50% width]                     │
# └────────────────────────────┴─────────────────────────────────┘
#
# KPI CARD DETAILS:
#   Total Loans:   [Total Loans], format #,##0
#   Total Volume:  [Total Loan Volume], format $#,##0,,.0B or $#,##0,,M
#   Default Rate:  [Overall Default Rate], format #.##"%"
#                  Delta color: Green <10%, Amber 10-15%, Red >15%
#   Avg FICO:      [Avg FICO Score], format #,##0
#   Expected Loss: [Expected Loss Per Loan], format $#,##0
#
# SLICERS:
#   Top-right: issue_year (dropdown multi-select)
#   Optional: grade (checklist)


# ================================================================
# PAGE 2: RISK DEEP DIVE
# ================================================================
#
# ┌──────────────────────────────────────────────────────────────┐
# │  RISK DEEP DIVE                   [Grade Slicer] [Term]     │
# ├──────────────────────────────┬───────────────────────────────┤
# │                              │                               │
# │  Default Rate by Sub-Grade   │  DTI vs Default Rate           │
# │  (Matrix / Heatmap visual)   │  (Grouped Bar Chart)           │
# │  Rows: sub_grade (A1-G5)     │  X: DTI buckets (use calc col) │
# │  Values: Default Rate        │  Y: Default Rate               │
# │  Conditional format:         │  Color: RAG by rate             │
# │    Green < 10%               │                                │
# │    Amber 10-20%              │  DTI Bucket = SWITCH(TRUE(),    │
# │    Red > 20%                 │    loans[dti]<10, "0-10",       │
# │                              │    loans[dti]<20, "10-20",      │
# │  [50% width]                 │    loans[dti]<30, "20-30",      │
# │                              │    "30+")                       │
# │                              │  [50% width]                    │
# ├──────────────────────────────┴───────────────────────────────┤
# │                                                              │
# │  Geographic Default Rate (Map or Bar by State)                │
# │  Location: loans[addr_state]                                  │
# │  Color saturation: Default Rate (green → red gradient)        │
# │  Tooltip: State, Loan Count, Default Rate, Avg FICO           │
# │  [Full width]                                                 │
# │                                                              │
# └──────────────────────────────────────────────────────────────┘
#
# CALCULATED COLUMN for DTI buckets:
# DTI Bucket = 
# SWITCH(
#     TRUE(),
#     loans[dti] < 10, "a. Low (0-10)",
#     loans[dti] < 20, "b. Moderate (10-20)",
#     loans[dti] < 30, "c. Elevated (20-30)",
#     loans[dti] < 40, "d. High (30-40)",
#     "e. Very High (40+)"
# )


# ================================================================
# PAGE 3: BORROWER RISK PROFILE & SCORECARD
# ================================================================
#
# ┌──────────────────────────────────────────────────────────────┐
# │  BORROWER RISK PROFILE                                      │
# ├──────────────────────────────┬───────────────────────────────┤
# │                              │                               │
# │  Defaulter vs Non-Defaulter  │  Feature Importance            │
# │  Profile Table               │  (embed charts/13 as image)    │
# │                              │  or re-create:                 │
# │  Columns:                    │  Horizontal Bar                │
# │    Metric | Non-Def | Def    │  Y: feature name               │
# │    Avg Loan                  │  X: abs coefficient             │
# │    Avg Rate                  │  Color: Red if positive,        │
# │    Avg DTI                   │         Green if negative       │
# │    Avg FICO                  │                                │
# │    Avg Income                │  [50% width]                   │
# │  [50% width]                 │                                │
# ├──────────────────────────────┴───────────────────────────────┤
# │                                                              │
# │  Risk Scorecard Visual (Table / Card layout)                  │
# │  ┌───────────────────────────────────────────────────┐        │
# │  │  BASE SCORE: 500                                  │        │
# │  │  ───────────────────────────────────────────────── │        │
# │  │  FICO 740+        → +40 pts    ⬇ Safer            │        │
# │  │  Grade A-B         → +35 pts    ⬇ Safer            │        │
# │  │  DTI < 15          → +25 pts    ⬇ Safer            │        │
# │  │  Grade F-G         → -40 pts    ⬆ Riskier          │        │
# │  │  Recent inquiries  → -20 pts    ⬆ Riskier          │        │
# │  │  ───────────────────────────────────────────────── │        │
# │  │  > 550: AUTO-APPROVE | 480-550: REVIEW | <480: DECLINE│   │
# │  └───────────────────────────────────────────────────┘        │
# │  (Build as a formatted multi-row card or text box)            │
# └──────────────────────────────────────────────────────────────┘
#
# NOTE: Scorecard values come from Sprint 2C output. Use the actual
# points printed by 03_credit_risk_modeling.py for accuracy.


# ================================================================
# PAGE 4: MODEL PERFORMANCE
# ================================================================
#
# ┌──────────────────────────────────────────────────────────────┐
# │  MODEL PERFORMANCE                                           │
# ├──────────┬──────────┬──────────┬──────────┬─────────────────┤
# │  ROC     │  PR      │   F1    │ Optimal  │  Default         │
# │  AUC     │  AUC     │  Score  │ Threshold│  Catch Rate      │
# ├──────────┴──────────┴──────────┴──────────┴─────────────────┤
# │                              │                               │
# │  ROC Curve                   │  Precision-Recall Curve        │
# │  (embed charts/11 as image)  │  (embed charts/12 as image)   │
# │  [50% width]                 │  [50% width]                   │
# │                              │                                │
# ├──────────────────────────────┴───────────────────────────────┤
# │                                                              │
# │  Confusion Matrix (embed charts/10 as image)                  │
# │  [50% width, centered]                                        │
# │                              │  Threshold Analysis             │
# │                              │  (embed charts/14 as image)    │
# │                              │  [50% width]                    │
# └──────────────────────────────┴───────────────────────────────┘
#
# NOTE: Since model charts are Python-generated PNGs, insert them
# using: Insert → Image → select from charts/ folder.
# This is standard practice for embedding ML output in Power BI.
#
# KPI values for model metrics cards — manually enter from the
# 03_credit_risk_modeling.py Executive Summary output:
#   ROC AUC:      (your value, e.g., 0.7103)
#   PR AUC:       (your value)
#   F1 Score:     (your value)
#   Threshold:    (your optimal value)
#   Catch Rate:   (TP / (TP+FN) at optimal threshold)


# ================================================================
# FORMATTING TIPS
# ================================================================
# 
# 1. All pages: consistent navy header bar (#1B4F72) at top with
#    white page title text
#
# 2. KPI cards: white background, subtle shadow, colored accent
#    left-border matching the metric meaning (green=good, red=bad)
#
# 3. Page navigation buttons at bottom of each page
#
# 4. Export final dashboard: File → Export → Export to PDF
#    Save as dashboard/credit_risk_dashboard.pdf for GitHub
