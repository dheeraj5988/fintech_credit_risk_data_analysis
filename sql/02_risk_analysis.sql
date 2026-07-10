-- ================================================================
-- Lending Club Credit Risk — SQL Risk Analytics
-- Sprint 2B: 12 Risk Queries
-- Author: Dheeraj Sharma
-- Database: credit_risk_analytics (MySQL) — table: loans
-- ================================================================
--
-- HOW TO RUN:
--   mysql -u root credit_risk_analytics < sql/02_risk_analysis.sql
--   (or copy-paste individual queries into MySQL terminal/Workbench)
--
-- Each query has:
--   [BUSINESS QUESTION] — the risk question being answered
--   [INTERPRETATION]    — what a credit risk officer concludes
-- ================================================================


-- ================================================================
-- SECTION 1: PORTFOLIO RISK METRICS
-- ================================================================

-- Q1: Overall Default Rate + Trend by Issue Year
-- ───────────────────────────────────────────────
-- [BUSINESS QUESTION] What is our portfolio-wide default rate, and
-- is loan quality improving or deteriorating over time (by vintage)?

SELECT
    issue_year,
    COUNT(*) AS loans_issued,
    SUM(`default`) AS defaults,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(SUM(loan_amnt) / 1000000, 1) AS total_issued_millions
FROM loans
GROUP BY issue_year
ORDER BY issue_year;

-- [INTERPRETATION] Note: recent years (2017-2018) show artificially
-- LOW default rates because those loans haven't had time to mature —
-- many are still "Current". This is called SEASONING BIAS, and
-- recognizing it is a mark of real credit risk understanding.
-- Compare only fully-seasoned vintages (2015 and earlier for 36-month
-- loans) for true quality comparison.


-- Q2: Default Rate by Grade with Volume (Risk Matrix)
-- ────────────────────────────────────────────────────
-- [BUSINESS QUESTION] How well does Lending Club's grading system
-- separate risk, and where is our loan volume concentrated?

SELECT
    grade,
    COUNT(*) AS loan_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM loans), 2) AS pct_of_portfolio,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_interest_rate,
    ROUND(AVG(loan_amnt), 0) AS avg_loan_amount,
    ROUND(AVG(fico_score), 0) AS avg_fico
FROM loans
GROUP BY grade
ORDER BY grade;

-- [INTERPRETATION] Default rate should rise monotonically A → G.
-- If int_rate rises alongside it, pricing is at least directionally
-- correct. The open question — answered in Q10 — is whether the
-- extra interest on risky grades actually COVERS the extra losses.


-- Q3: Expected Loss by Grade
-- ──────────────────────────
-- [BUSINESS QUESTION] For every dollar lent in each grade, how much
-- do we expect to lose? (Core credit risk metric: EL = PD × EAD × LGD)
--
-- PD  = Probability of Default (our default rate)
-- EAD = Exposure At Default (approximated by avg loan amount)
-- LGD = Loss Given Default (industry standard assumption: ~90% for
--       unsecured personal loans after recovery costs)

SELECT
    grade,
    COUNT(*) AS loan_count,
    ROUND(AVG(`default`), 4) AS pd_probability_of_default,
    ROUND(AVG(loan_amnt), 0) AS avg_exposure,
    0.90 AS lgd_assumption,
    ROUND(AVG(`default`) * AVG(loan_amnt) * 0.90, 0) AS expected_loss_per_loan,
    ROUND(AVG(`default`) * 0.90 * 100, 2) AS expected_loss_pct_of_principal
FROM loans
GROUP BY grade
ORDER BY grade;

-- [INTERPRETATION] Grade G loans lose ~25%+ of principal in
-- expectation. For the portfolio to be profitable, the interest
-- rate must exceed expected_loss_pct + funding cost + operating
-- cost. This is exactly how banks price credit.


-- Q4: Concentration Risk — Portfolio Dollars by Grade
-- ────────────────────────────────────────────────────
-- [BUSINESS QUESTION] Are we overexposed to risky grades in DOLLAR
-- terms (not just loan counts)?

SELECT
    grade,
    ROUND(SUM(loan_amnt) / 1000000, 1) AS total_exposure_millions,
    ROUND(SUM(loan_amnt) * 100.0 / (SELECT SUM(loan_amnt) FROM loans), 2) AS pct_of_dollar_portfolio,
    ROUND(SUM(loan_amnt * `default`) / 1000000, 1) AS defaulted_dollars_millions
FROM loans
GROUP BY grade
ORDER BY grade;

-- [INTERPRETATION] If B and C grades hold 50%+ of dollar exposure,
-- the portfolio's health depends heavily on mid-grade performance —
-- a recession that pushes C-grade defaults from 15% → 25% would hit
-- harder than G-grade doubling, purely due to volume.


-- ================================================================
-- SECTION 2: BORROWER RISK PROFILING
-- ================================================================

-- Q5: DTI Risk Buckets
-- ────────────────────
-- [BUSINESS QUESTION] At what DTI level does default risk
-- meaningfully accelerate? (Sets underwriting cutoffs)

SELECT
    CASE
        WHEN dti < 10 THEN 'a. Low (0-10)'
        WHEN dti < 20 THEN 'b. Moderate (10-20)'
        WHEN dti < 30 THEN 'c. Elevated (20-30)'
        WHEN dti < 40 THEN 'd. High (30-40)'
        ELSE 'e. Very High (40+)'
    END AS dti_bucket,
    COUNT(*) AS loan_count,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_int_rate,
    ROUND(AVG(annual_inc_capped), 0) AS avg_income
FROM loans
GROUP BY dti_bucket
ORDER BY dti_bucket;

-- [INTERPRETATION] Look for the "knee" in the curve — the bucket
-- where default rate jumps disproportionately. That's where an
-- underwriting policy would draw the line (e.g., "DTI > 35 requires
-- manual review or higher pricing").


-- Q6: Defaulter vs Non-Defaulter Profile (Side-by-Side CTE)
-- ──────────────────────────────────────────────────────────
-- [BUSINESS QUESTION] What does the "average defaulter" look like
-- versus the "average good borrower"?

WITH profiles AS (
    SELECT
        `default`,
        AVG(loan_amnt) AS avg_loan,
        AVG(int_rate) AS avg_rate,
        AVG(dti) AS avg_dti,
        AVG(annual_inc_capped) AS avg_income,
        AVG(fico_score) AS avg_fico,
        AVG(revol_util) AS avg_revol_util,
        AVG(inq_last_6mths) AS avg_inquiries,
        AVG(delinq_2yrs) AS avg_delinq,
        COUNT(*) AS n
    FROM loans
    GROUP BY `default`
)
SELECT
    CASE WHEN `default` = 1 THEN 'Defaulter' ELSE 'Good Borrower' END AS profile,
    n AS count,
    ROUND(avg_loan, 0) AS avg_loan_amount,
    ROUND(avg_rate, 2) AS avg_interest_rate,
    ROUND(avg_dti, 1) AS avg_dti,
    ROUND(avg_income, 0) AS avg_annual_income,
    ROUND(avg_fico, 0) AS avg_fico_score,
    ROUND(avg_revol_util, 1) AS avg_revolving_utilization,
    ROUND(avg_inquiries, 2) AS avg_credit_inquiries_6m,
    ROUND(avg_delinq, 3) AS avg_delinquencies_2yr
FROM profiles
ORDER BY `default`;

-- [INTERPRETATION] Defaulters typically show: higher rate (LC knew),
-- higher DTI, lower income, lower FICO, higher revolving utilization,
-- and more recent credit inquiries (credit-seeking behavior is a
-- classic pre-default signal).


-- Q7: Employment Length vs Default (with Volume Filter)
-- ──────────────────────────────────────────────────────
-- [BUSINESS QUESTION] Does employment stability actually predict
-- repayment? And what about borrowers who didn't report employment?

SELECT
    CASE
        WHEN emp_length_missing = 1 THEN 'Not Reported'
        WHEN emp_length_years = 0 THEN '< 1 year'
        WHEN emp_length_years BETWEEN 1 AND 3 THEN '1-3 years'
        WHEN emp_length_years BETWEEN 4 AND 6 THEN '4-6 years'
        WHEN emp_length_years BETWEEN 7 AND 9 THEN '7-9 years'
        ELSE '10+ years'
    END AS employment_bucket,
    COUNT(*) AS loan_count,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(AVG(annual_inc_capped), 0) AS avg_income
FROM loans
GROUP BY employment_bucket
HAVING loan_count >= 1000
ORDER BY default_rate_pct DESC;

-- [INTERPRETATION] "Not Reported" typically has the HIGHEST default
-- rate — missing data is itself a risk signal. This validates our
-- cleaning decision to flag missing emp_length rather than impute
-- it away.


-- Q8: Income-to-Loan Ratio Quartiles by Outcome
-- ──────────────────────────────────────────────
-- [BUSINESS QUESTION] Does borrowing a large amount relative to
-- income predict default? (Payment burden proxy)

WITH ratio_calc AS (
    SELECT
        `default`,
        annual_inc_capped / NULLIF(loan_amnt, 0) AS income_to_loan,
        NTILE(4) OVER (ORDER BY annual_inc_capped / NULLIF(loan_amnt, 0)) AS ratio_quartile
    FROM loans
    WHERE annual_inc_capped > 0
)
SELECT
    CONCAT('Q', ratio_quartile) AS income_to_loan_quartile,
    CASE ratio_quartile
        WHEN 1 THEN 'Lowest income vs loan (most stretched)'
        WHEN 2 THEN 'Below median'
        WHEN 3 THEN 'Above median'
        WHEN 4 THEN 'Highest income vs loan (least stretched)'
    END AS description,
    COUNT(*) AS loan_count,
    ROUND(AVG(income_to_loan), 1) AS avg_income_to_loan_ratio,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct
FROM ratio_calc
GROUP BY ratio_quartile
ORDER BY ratio_quartile;

-- [INTERPRETATION] Q1 borrowers (borrowing the most relative to
-- income) should default significantly more than Q4. This justifies
-- the income_to_loan_ratio feature we engineer for the model.


-- Q9: Loan Vintage Analysis — Default by Issue Quarter
-- ─────────────────────────────────────────────────────
-- [BUSINESS QUESTION] Which origination periods produced the worst
-- loans? (Vintage analysis is a core credit risk discipline)

SELECT
    issue_year,
    QUARTER(issue_date) AS issue_quarter,
    COUNT(*) AS loans_issued,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(AVG(int_rate), 2) AS avg_rate_charged
FROM loans
WHERE issue_year BETWEEN 2012 AND 2015   -- fully seasoned vintages only
GROUP BY issue_year, issue_quarter
ORDER BY issue_year, issue_quarter;

-- [INTERPRETATION] Rising default rates across consecutive vintages
-- with FLAT interest rates = underwriting standards loosening without
-- pricing compensation — the classic pattern before credit losses
-- spike. LC's 2015-2016 vintages are known in industry for exactly this.


-- ================================================================
-- SECTION 3: ADVANCED RISK-RETURN ANALYSIS
-- ================================================================

-- Q10: Risk-Adjusted Return by Grade
-- ───────────────────────────────────
-- [BUSINESS QUESTION] After expected losses, which grade actually
-- delivers the best return? (The money question)
--
-- Simplified model:
--   Gross return ≈ avg interest rate
--   Expected loss ≈ default_rate × LGD (90%)
--   Net risk-adjusted return ≈ gross - expected loss

SELECT
    grade,
    COUNT(*) AS loan_count,
    ROUND(AVG(int_rate), 2) AS gross_return_pct,
    ROUND(AVG(`default`) * 90, 2) AS expected_loss_pct,
    ROUND(AVG(int_rate) - AVG(`default`) * 90, 2) AS net_risk_adjusted_return_pct
FROM loans
GROUP BY grade
ORDER BY net_risk_adjusted_return_pct DESC;

-- [INTERPRETATION] Typically B/C grades win — A is safe but low-yield,
-- and F/G interest rates DON'T fully compensate for their losses.
-- An investor should overweight B/C. This single query demonstrates
-- portfolio thinking that most fresher candidates never show.


-- Q11: The Verification Paradox
-- ─────────────────────────────
-- [BUSINESS QUESTION] Do income-verified loans default LESS than
-- unverified ones? (The answer surprises most people)

SELECT
    verification_status,
    COUNT(*) AS loan_count,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct,
    ROUND(AVG(annual_inc_capped), 0) AS avg_income,
    ROUND(AVG(dti), 1) AS avg_dti,
    ROUND(AVG(int_rate), 2) AS avg_int_rate
FROM loans
GROUP BY verification_status
ORDER BY default_rate_pct DESC;

-- [INTERPRETATION] Verified loans usually show HIGHER default rates.
-- Why? SELECTION BIAS — LC requires verification for applications
-- that already look risky. Verification is a CONSEQUENCE of risk,
-- not a cause of safety. Explaining this correctly in an interview
-- demonstrates statistical maturity (confounding/selection effects).


-- Q12: FICO Band × Grade Cross-Analysis
-- ──────────────────────────────────────
-- [BUSINESS QUESTION] Within the same grade, does FICO still add
-- risk separation? (Tests whether grade fully captures credit score)

SELECT
    grade,
    CASE
        WHEN fico_score < 670 THEN 'a. Fair (<670)'
        WHEN fico_score < 740 THEN 'b. Good (670-739)'
        ELSE 'c. Very Good (740+)'
    END AS fico_band,
    COUNT(*) AS loan_count,
    ROUND(AVG(`default`) * 100, 2) AS default_rate_pct
FROM loans
WHERE grade IN ('B', 'C', 'D')     -- focus on the high-volume middle grades
GROUP BY grade, fico_band
HAVING loan_count >= 500
ORDER BY grade, fico_band;

-- [INTERPRETATION] If default rates differ across FICO bands WITHIN
-- the same grade, FICO adds signal beyond grade — justification for
-- including both in the model rather than assuming grade absorbs
-- everything.


-- ================================================================
-- END OF SPRINT 2B SQL — 12 RISK QUERIES COMPLETE
-- ================================================================
--
-- CREDIT RISK CONCEPTS COVERED:
--   1. Seasoning bias / vintage maturity (Q1)
--   2. Risk grading validation (Q2)
--   3. Expected Loss = PD × EAD × LGD (Q3)
--   4. Concentration risk in dollar terms (Q4)
--   5. Underwriting cutoff analysis (Q5)
--   6. Borrower risk profiling (Q6)
--   7. Missing data as a risk signal (Q7)
--   8. Payment burden ratios (Q8)
--   9. Vintage analysis (Q9)
--  10. Risk-adjusted return (Q10)
--  11. Selection bias in observational data (Q11)
--  12. Marginal signal analysis (Q12)
--
-- NEXT: run scripts/02_feature_engineering.py, then Sprint 2C modeling
