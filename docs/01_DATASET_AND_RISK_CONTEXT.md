# Dataset & Risk Context — Lending Club Loan Data

## 1. Download the Dataset

**Source:** Kaggle — [Lending Club Loan Data (2007-2018)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Steps:
1. Go to the link above (Kaggle account required)
2. Click **Download** — this is a LARGE dataset (~1.5GB zipped, ~2.9GB unzipped)
3. Unzip it — you'll get `accepted_2007_to_2018Q4.csv.gz` (and a rejected loans file we won't use)
4. Place `accepted_2007_to_2018Q4.csv.gz` into `data/raw/`

**Note on size:** This file has ~2.26 million rows and 151 columns. Our cleaning
script reads it in a memory-efficient way and will take a few minutes to run.
If your machine struggles, the script has a `SAMPLE_FRAC` setting you can
lower (see comments in `01_clean_and_eda.py`).

---

## 2. Data Dictionary — Critical Columns (Banking Context)

This is what a credit risk analyst actually cares about. Understanding
WHY each column matters is what separates "I ran a model" from "I did
risk analysis" in an interview.

| Column | Meaning | Why It Matters for Risk |
|---|---|---|
| `loan_amnt` | Amount borrower requested | Larger loans = larger potential loss (higher severity) |
| `term` | Loan length (36 or 60 months) | Longer terms = more time for borrower's situation to change = higher risk |
| `int_rate` | Interest rate charged | Lending Club's OWN risk assessment — higher rate = they already think it's riskier |
| `installment` | Monthly payment amount | High installment relative to income = payment shock risk |
| `grade` / `sub_grade` | LC's risk grade (A=safest, G=riskiest) | The single strongest predictor of default in this dataset |
| `emp_length` | Years employed | Employment stability signal; <1 year = income risk |
| `home_ownership` | RENT / OWN / MORTGAGE | Owning correlates with financial stability |
| `annual_inc` | Self-reported annual income | Repayment capacity — but self-reported, so `verification_status` matters |
| `verification_status` | Was income verified? | Unverified income is a red flag — self-reported numbers are often inflated |
| `loan_status` | Current loan outcome | **This is our target variable** — see engineering below |
| `purpose` | Why the loan was taken | Debt consolidation loans behave differently than small_business loans (much riskier) |
| `dti` | Debt-to-Income ratio | THE core underwriting metric — high DTI = borrower is already stretched thin |
| `delinq_2yrs` | Past delinquencies (2yr) | Past behavior predicts future behavior — strong signal |
| `fico_range_low/high` | Credit score range | Traditional credit risk score, still highly predictive |
| `open_acc` | Number of open credit lines | Too many open lines = potential overextension |
| `pub_rec` | Public derogatory records (bankruptcies etc.) | Major red flag if > 0 |
| `revol_util` | Revolving credit utilization % | High utilization (close to credit limit) = financial stress signal |
| `total_acc` | Total credit accounts ever opened | Credit history depth |
| `mort_acc` | Number of mortgage accounts | Additional stability/leverage signal |

---

## 3. Target Variable Engineering

`loan_status` has multiple values. We collapse it into a binary target:

**Default = 1** (bad outcome):
- `Charged Off`
- `Default`
- `Late (31-120 days)`

**Non-Default = 0** (good outcome):
- `Fully Paid`
- `Current`

**Excluded from modeling** (ambiguous/in-progress, not yet resolved):
- `In Grace Period`
- `Late (16-30 days)`
- `Issued`

This is a standard industry approach — you can't label a loan that hasn't
reached a resolved outcome yet.

---

## 4. CRITICAL: Data Leakage Columns (Must Be Removed)

These columns only exist or get populated AFTER the loan outcome is known.
Including them would make your model "predict" default using information
that wasn't available at the time the loan was approved — this is the
#1 mistake that gets junior analysts and data scientists corrected in
interviews and code reviews.

**Remove these before any modeling:**
```
total_pymnt, total_pymnt_inv, total_rec_prncp, total_rec_int,
total_rec_late_fee, recoveries, collection_recovery_fee,
last_pymnt_d, last_pymnt_amnt, next_pymnt_d, last_credit_pull_d,
out_prncp, out_prncp_inv, hardship_flag, hardship_type,
hardship_reason, hardship_status, hardship_amount,
debt_settlement_flag, settlement_status, settlement_amount,
settlement_date, funded_amnt, funded_amnt_inv
```

`funded_amnt` and `funded_amnt_inv` are technically known at issuance but
are near-duplicates of `loan_amnt` (funding is rarely partial) — we drop
them to avoid redundancy, not leakage.

---

## 5. Next Step

Once the CSV is in `data/raw/`, run `scripts/01_clean_and_eda.py`
(see README.md for exact command).
