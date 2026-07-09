# Workforce Cost & Attrition Risk Model
### A Zero-to-One People Analytics Build

A end-to-end people-analytics system that integrates fragmented HR, payroll, and
market-benchmark data into one governed dataset, predicts which employees are at
risk of leaving, and translates that risk into a clear financial decision for
leadership: **what does it cost to act, versus what does it cost to do nothing?**

This project was built to mirror a real HR Data & AI Analyst brief — integrating
Dayforce-style HRIS, finance/payroll, and external salary-benchmarking sources;
building attrition risk and compensation scenario models; and delivering an
executive dashboard for non-technical leadership and PE stakeholders.

> All data is **synthetic** and generated locally. No real employee data is used.

---

## The headline finding (plain English)

On a workforce of **524 active employees** with a modelled annual attrition rate of
**~23%**, the model identifies a **retention watchlist of 64 people** (the top 12%
by risk), who sit on average **~12% below market pay**.

Over a 24-month horizon, three paths were costed:

| Scenario | What it means | 24-month cost |
|---|---|---|
| **B. Proactive Retention** | Targeted raises for the watchlist now | **~€106.1M** |
| **A. Status Quo** | No change; replace leavers as they go | ~€106.5M |
| **C. Reactive (Do Nothing)** | Replace leavers *plus* eat the productivity gap | **~€111.7M** |

**The decision in one line:** spending **~€0.68M** on targeted retention raises is
projected to **save ~€0.43M against status quo** and **avoid ~€5.6M** versus letting
attrition run its course and backfilling reactively.

*(Exact figures shift slightly if the synthetic data is regenerated with a
different random seed; the structure and direction of the result are stable.)*

---

## What's in the box

```
workforce_analytics/
├── README.md                          ← you are here
├── src/
│   ├── 01_generate_synthetic_data.py  ← builds 3 messy source files
│   ├── 02_etl_integration.py          ← integrates + governs into one dataset
│   ├── 03_attrition_model.py          ← scikit-learn risk model + watchlist
│   ├── 04_compensation_scenario_model.py ← 3-scenario cost model + Excel export
│   └── 05_build_dashboard.py          ← interactive executive dashboard
├── data/
│   ├── raw/                           ← hris_dayforce / payroll_export / market_benchmark
│   └── processed/master_workforce.csv ← the governed, unified dataset
├── models/feature_importance.csv
└── outputs/
    ├── data_quality_report.txt        ← every issue found + how it was fixed
    ├── scored_active_employees.csv     ← risk score 0–100 for every active employee
    ├── retention_watchlist.csv         ← the actionable shortlist
    ├── model_evaluation.txt            ← metrics, in plain language
    ├── scenario_summary.csv
    ├── compensation_scenario_model.xlsx ← live-formula Excel model
    └── executive_dashboard.html        ← open in any browser, no install needed
```

### To run it yourself
```bash
pip install pandas numpy scikit-learn openpyxl plotly
cd workforce_analytics
python src/01_generate_synthetic_data.py
python src/02_etl_integration.py
python src/03_attrition_model.py
python src/04_compensation_scenario_model.py
python src/05_build_dashboard.py
# then open outputs/executive_dashboard.html
```
Each script is self-contained and reads the outputs of the previous step.

---

## How it works, step by step

### 1. Synthetic data — built with a *real* signal to find
Rather than random noise, attrition is generated from a latent "flight-risk"
propensity that encodes findings consistent with real people-analytics research:
people who are **paid below market**, who **haven't had a raise in a long time**,
who sit **low in their salary band**, or who are **very early or very late in
tenure** are more likely to leave. Gaussian noise is layered on so the relationship
is genuine but **not perfectly separable** — which is why the model lands at a
realistic ROC-AUC in the mid-0.80s rather than a suspicious 0.99.

The data is deliberately **split across three source files with different employee-ID
formats** (HRIS uses `DF-100001`, payroll uses `500001`, the benchmark vendor uses
`EMP_00001` and only covers ~88% of staff). This forces genuine integration logic.

### 2. Data integration & governance
The ETL layer joins HRIS → payroll (on a shared tax reference) → benchmark (on the
HRIS link), then runs a battery of governance checks. Every issue is logged with a
severity and the remediation applied. On a typical run it catches and fixes:

- a **duplicate** payroll record (double export)
- **5 missing salaries** (imputed with department × level median)
- **1 out-of-range salary** (a 10× data-entry error, capped to band)
- **8 missing performance ratings** (median-imputed, flagged)
- **6 departments with inconsistent casing/whitespace** (canonicalised)
- a **12% external-benchmark coverage gap** (imputed from internal band mid)

Crucially, every imputed or corrected value keeps an audit flag
(`salary_imputed`, `benchmark_imputed`, etc.) so nothing is silently changed. See
`outputs/data_quality_report.txt`.

### 3. Attrition risk model
Two models are trained and compared honestly on a held-out test set: **Logistic
Regression** (interpretable, calibrated) and **Random Forest** (captures the
U-shaped tenure effect). The better performer is selected on test ROC-AUC — both
land around **0.85**.

- No leakage: exit dates and status are never used as features.
- Class imbalance handled with balanced class weights.
- Output is a **0–100 risk score** for every active employee, not just a yes/no.
- An **early-warning operating point** is reported — tuned to catch ~70–75% of
  real leavers — which is how an HR team would actually run a finite intervention
  budget.
- The top **12% highest-risk** active employees become the **watchlist**.

The strongest *actionable* drivers are compensation-related (pay vs market, time
since last raise, position in band) — levers leadership can actually pull.

### 4. Compensation cost scenario model
Replacement cost is modelled as a **multiple of salary scaled by seniority**
(0.5× for juniors up to 2.0× for directors), bundling recruiting, onboarding, ramp,
and lost productivity — consistent with widely-cited industry ranges. The model
projects total workforce cost across the three scenarios above and exports a
**fully formula-driven Excel workbook** (Assumptions / Employee Detail / Scenario
Summary). Every assumption is a blue input cell; change one and the whole model
recomputes. The workbook was validated to contain **zero formula errors** across
~3,150 formulas.

### 5. Executive dashboard
A single self-contained `executive_dashboard.html` (Plotly is embedded, so it works
**offline** — safe to demo live on any machine). It leads with the financial
decision (cost of inaction vs intervention), shows where risk concentrates by
department, what's driving it, and the named retention watchlist — all framed in
euros, not model jargon.

---

## Methodology notes & assumptions (for the technical reader)

- **Horizon scaling:** an annual churn probability `p` is converted to a 24-month
  probability via `1 − (1 − p)^(24/12)`, applied identically in Python and in the
  Excel model so the two reconcile exactly.
- **Risk → probability:** risk scores are the model's predicted probabilities ×100.
  They are rank-meaningful; the watchlist uses a percentile cut because that's how a
  fixed intervention budget is actually allocated.
- **Retention-raise effect:** an 8% targeted raise is assumed to cut a watchlist
  employee's churn probability by 45%. This is the single most sensitive assumption
  and is exposed as an editable input — a reviewer can stress-test it directly.
- **What this is *not*:** a causal model. It identifies who is *at risk* and what
  acting *would cost*; it does not prove a raise causes retention. The scenario
  model makes that assumption explicit and adjustable rather than hiding it.

---

## How I'd describe this in an interview (30 seconds)

> "I built an end-to-end people-analytics system that takes three disconnected,
> deliberately messy data sources — HRIS, payroll, and external salary benchmarks
> with mismatched IDs — and integrates them into one governed dataset with a full
> data-quality audit trail. On top of that I trained an attrition risk model that
> scores every active employee and flags a retention watchlist, then I wrapped it
> in a compensation scenario model that turns those risk scores into a board-level
> financial decision: it shows that roughly €0.7M of targeted retention raises
> avoids around €5.6M of replacement and productivity cost over two years. The whole
> thing ends in an interactive executive dashboard a CFO or Head of HR can read in
> euros, not risk scores."

---

*Built as a portfolio project. Synthetic data only. Model performance and exact euro
figures are representative of a single generated run and will vary slightly on
regeneration; the methodology and the direction of the conclusions are stable.*
