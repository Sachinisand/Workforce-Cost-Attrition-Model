"""
01_generate_synthetic_data.py
================================
Generates a realistic, internally-consistent synthetic workforce dataset and
splits it across THREE source files that mimic the real-world integration
problem this project is built around:

    - hris_dayforce.csv      (HR system of record: org structure, dates, ratings)
    - payroll_export.csv     (Finance/payroll: actual paid compensation)
    - market_benchmark.csv   (External salary benchmarking vendor)

Each source uses a DIFFERENT employee-ID format on purpose, so the ETL layer
(02_etl_integration.py) has to do genuine reconciliation work, not a trivial join.

DESIGN OF THE ATTRITION SIGNAL
------------------------------
Attrition is NOT random. We build a latent "flight risk" propensity from drivers
that mirror real people-analytics findings, then sample actual leavers from it:

    higher risk when:  paid below market band (comp_ratio < 1)
                       long time since last merit increase
                       low performance rating BUT under-paid (frustrated performers)
                       early tenure (0-18 months) and late tenure plateau
                       certain high-churn departments (Sales, Support)
    lower risk when:   paid at/above market, recent raise, mid-tenure, high level

Gaussian noise is added to the latent score so the relationship is real but not
perfectly separable — a model should land around 0.80-0.88 ROC-AUC, which is
realistic and defensible in an interview (not a suspicious 0.99).

Run:  python src/01_generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)          # reproducible
N_EMPLOYEES = 680
AS_OF_DATE = datetime(2026, 1, 1)        # snapshot date for tenure / "active" status
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Reference structures: departments, levels, locations, salary bands
# ---------------------------------------------------------------------------
# Salary bands are defined per (department, job_level). mid is the market anchor;
# min/max are +/- ~18% around mid. These bands are the SAME ones the benchmark
# vendor reports against, so comp-to-band is meaningful.
DEPARTMENTS = {
    # dept: (relative pay weight, base churn propensity on latent scale)
    "Engineering":      (1.25, -0.15),
    "Product":          (1.20, -0.10),
    "Data & Analytics": (1.18, -0.15),
    "Sales":            (1.10, 0.55),   # high churn
    "Customer Success": (0.95, 0.45),   # high churn
    "Support":          (0.80, 0.70),   # highest churn
    "Marketing":        (0.95, 0.10),
    "Finance":          (1.00, -0.20),
    "People/HR":        (0.92, -0.20),
    "Operations":       (0.90, 0.00),
}

JOB_LEVELS = {
    # level: (base_salary_anchor_EUR, relative attrition modifier)
    "L1 - Junior":      (48000, 0.20),
    "L2 - Mid":         (66000, 0.05),
    "L3 - Senior":      (88000, -0.05),
    "L4 - Lead":        (112000, -0.10),
    "L5 - Manager":     (130000, -0.15),
    "L6 - Director":    (165000, -0.20),
}

LOCATIONS = {
    "Munich":    1.05,
    "Hamburg":   1.00,
    "Berlin":    0.98,
    "Remote-DE": 0.96,
    "London":    1.15,
    "Lisbon":    0.82,
}

LEVEL_ORDER = list(JOB_LEVELS.keys())

# ---------------------------------------------------------------------------
# 2. Generate the "true" master records (we'll split them afterwards)
# ---------------------------------------------------------------------------
records = []
for i in range(N_EMPLOYEES):
    _dept_w = np.array([1, 0.6, 0.8, 1.3, 0.9, 1.1, 0.7, 0.6, 0.5, 0.8])
    dept = RNG.choice(list(DEPARTMENTS.keys()), p=_dept_w / _dept_w.sum())
    dept_pay_w, dept_churn = DEPARTMENTS[dept]

    # Level distribution skews junior/mid (realistic pyramid)
    level = RNG.choice(LEVEL_ORDER, p=[0.28, 0.30, 0.22, 0.10, 0.07, 0.03])
    level_anchor, level_churn = JOB_LEVELS[level]

    location = RNG.choice(list(LOCATIONS.keys()),
                          p=[0.22,0.18,0.15,0.20,0.10,0.15])
    loc_w = LOCATIONS[location]

    # --- Salary band for this dept/level (mid = market anchor) ---
    band_mid = level_anchor * dept_pay_w * loc_w
    band_mid *= RNG.normal(1.0, 0.02)            # tiny structural noise
    band_min = band_mid * 0.82
    band_max = band_mid * 1.18

    # --- Actual base salary: where the person actually sits in/around band ---
    # Wider spread so a meaningful share of people are genuinely under/over market,
    # giving the model real comp signal to learn from.
    position_in_band = np.clip(RNG.normal(0.50, 0.28), 0.0, 1.1)
    base_salary = band_min + position_in_band * (band_max - band_min)

    # --- Market benchmark: external vendor's view of the going rate ---
    # Anchored to the band mid (the going market rate for the role), with vendor
    # noise. Because individuals sit at different points in band, some are
    # genuinely below market and some above — an INDIVIDUAL property, not a
    # structural department artifact. This keeps comp_ratio a clean signal.
    market_benchmark = band_mid * RNG.normal(1.0, 0.07)

    # --- Tenure & hire date ---
    tenure_months = int(np.clip(RNG.gamma(shape=2.2, scale=18), 1, 220))
    hire_date = AS_OF_DATE - timedelta(days=int(tenure_months * 30.4))

    # --- Last merit increase ---
    # months_since_last_raise correlates with tenure but capped; some never raised.
    months_since_raise = int(np.clip(RNG.normal(min(tenure_months, 14), 6), 0, 48))
    if tenure_months < 6:
        months_since_raise = tenure_months  # too new to have had a raise
    last_merit_increase_pct = round(float(np.clip(RNG.normal(4.2, 2.0), 0, 12)), 2)
    if months_since_raise > 18:
        last_merit_increase_pct = round(last_merit_increase_pct * 0.4, 2)  # stale

    # --- Performance rating 1..5 (skewed toward 3-4) ---
    performance_rating = int(RNG.choice([1,2,3,4,5], p=[0.04,0.12,0.40,0.32,0.12]))

    records.append({
        "_idx": i,
        "department": dept,
        "job_level": level,
        "location": location,
        "hire_date": hire_date.date().isoformat(),
        "tenure_months": tenure_months,
        "performance_rating": performance_rating,
        "salary_band_min": round(band_min, 0),
        "salary_band_mid": round(band_mid, 0),
        "salary_band_max": round(band_max, 0),
        "market_benchmark_salary": round(market_benchmark, 0),
        "base_salary": round(base_salary, 0),
        "months_since_last_raise": months_since_raise,
        "last_merit_increase_pct": last_merit_increase_pct,
        "_dept_churn": dept_churn,
        "_level_churn": level_churn,
    })

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 3. Build the latent flight-risk score and sample actual leavers
# ---------------------------------------------------------------------------
comp_ratio = df["base_salary"] / df["market_benchmark_salary"]   # <1 => underpaid
band_penetration = (df["base_salary"] - df["salary_band_min"]) / \
                   (df["salary_band_max"] - df["salary_band_min"])

# Tenure risk: U-shaped — risky when very new (<18m) and again at long plateau (>72m)
tenure = df["tenure_months"].values
tenure_risk = np.where(tenure < 18, 0.9 - tenure/40,
                np.where(tenure > 72, 0.2 + (tenure-72)/300, -0.1))

# Frustrated-performer effect: strong performers who are underpaid leave most
frustrated = ((df["performance_rating"] >= 4) & (comp_ratio < 0.95)).astype(float) * 0.8

# Signal scale lifts the latent spread so the relationship is strong enough to
# learn (target model ROC-AUC ~0.80-0.84) while the noise term keeps it realistic
# and not perfectly separable.
SIGNAL = 1.6
latent = SIGNAL * (
    5.5 * (1 - comp_ratio)                        # underpaid vs market -> up (dominant)
    + 1.8 * (df["months_since_last_raise"] / 24)   # stale raise -> up
    - 1.4 * band_penetration                       # high in band -> down
    + 1.1 * tenure_risk
    + 1.0 * frustrated
    + df["_dept_churn"]
    + df["_level_churn"]
    - 0.18 * (df["performance_rating"] - 3)         # low performers slightly more
) + RNG.normal(0, 0.45, len(df))                    # IRREDUCIBLE NOISE

# Convert latent -> probability via a sigmoid with a calibrated intercept so the
# overall attrition rate lands ~22-25%. We deliberately AVOID post-hoc rescaling,
# which would compress the score range and destroy the signal.
latent = latent - latent.mean()
INTERCEPT = -2.10   # tuned so mean(prob) ~ 0.23
prob = 1 / (1 + np.exp(-(latent + INTERCEPT)))
prob = np.clip(prob, 0.01, 0.96)
df["attrition_flag"] = (RNG.random(len(df)) < prob).astype(int)

# Exit dates for leavers (within last 18 months, must be after hire)
exit_dates = []
for _, r in df.iterrows():
    if r["attrition_flag"] == 1:
        max_days_ago = min(540, max(30, r["tenure_months"] * 30 // 2))
        days_ago = RNG.integers(15, max_days_ago + 1)
        exit_dates.append((AS_OF_DATE - timedelta(days=int(days_ago))).date().isoformat())
    else:
        exit_dates.append("")
df["exit_date"] = exit_dates

print(f"Generated {len(df)} employees | overall attrition rate: "
      f"{df['attrition_flag'].mean():.1%}")
print("Attrition by department:")
print((df.groupby('department')['attrition_flag'].mean().sort_values(ascending=False)
       .apply(lambda x: f'{x:.1%}')).to_string())

# ---------------------------------------------------------------------------
# 4. Assign manager_id (within department, a higher-level employee)
# ---------------------------------------------------------------------------
df["manager_idx"] = -1
for dept in df["department"].unique():
    dept_df = df[df["department"] == dept]
    leaders = dept_df[dept_df["job_level"].isin(
        ["L4 - Lead", "L5 - Manager", "L6 - Director"])]
    if len(leaders) == 0:
        leaders = dept_df.nlargest(2, "tenure_months")
    for idx in dept_df.index:
        if idx in leaders.index and len(leaders) > 1:
            mgr = leaders[leaders.index != idx].sample(1, random_state=int(idx)).iloc[0]
        else:
            mgr = leaders.sample(1, random_state=int(idx)).iloc[0]
        df.at[idx, "manager_idx"] = mgr["_idx"]

# ---------------------------------------------------------------------------
# 5. Split into THREE source files with DIFFERENT ID formats
# ---------------------------------------------------------------------------
# Master canonical key is the row index. Each source derives its own ID format.
#   HRIS (Dayforce):  "DF-100001" style
#   Payroll:          numeric employee number "500001"
#   Benchmark vendor: "EMP_00001" style (and only covers a SUBSET — vendors never
#                     have 100% coverage, forcing the ETL to handle missing matches)

df["hris_id"]    = df["_idx"].apply(lambda x: f"DF-{100000 + x}")
df["payroll_id"] = df["_idx"].apply(lambda x: str(500000 + x))
df["bench_id"]   = df["_idx"].apply(lambda x: f"EMP_{x:05d}")

# A crosswalk the ETL must rebuild via a shared natural key.
# We expose a shared "national_insurance"-like token on HRIS & payroll, and the
# benchmark file matches on (department, job_level, location) + a vendor token.
# To keep it tractable but non-trivial: HRIS<->Payroll share `tax_ref`,
# Benchmark shares `bench_id` that maps to hris via a provided mapping file.
df["tax_ref"] = df["_idx"].apply(lambda x: f"DE{93000000 + x*7 % 9999999:08d}")

# ---- 5a. HRIS / Dayforce source ----
hris = pd.DataFrame({
    "Employee ID": df["hris_id"],
    "Tax Reference": df["tax_ref"],
    "Department": df["department"],
    "Job Level": df["job_level"],
    "Location": df["location"],
    "Hire Date": df["hire_date"],
    "Tenure (Months)": df["tenure_months"],
    "Performance Rating": df["performance_rating"],
    "Manager ID": df["manager_idx"].apply(
        lambda x: f"DF-{100000 + int(x)}" if x >= 0 else ""),
    "Status": np.where(df["attrition_flag"] == 1, "Terminated", "Active"),
    "Exit Date": df["exit_date"],
})

# ---- 5b. Payroll source (numeric IDs, paid comp, joins via Tax Reference) ----
payroll = pd.DataFrame({
    "emp_no": df["payroll_id"],
    "tax_ref": df["tax_ref"],
    "annual_base_eur": df["base_salary"],
    "band_min": df["salary_band_min"],
    "band_mid": df["salary_band_mid"],
    "band_max": df["salary_band_max"],
    "last_increase_pct": df["last_merit_increase_pct"],
    "months_since_increase": df["months_since_last_raise"],
})

# ---- 5c. External benchmark source (subset coverage, EMP_ ids) ----
# Vendor only benchmarks ~88% of roles; missing ones must be handled in ETL.
covered = df.sample(frac=0.88, random_state=7).copy()
benchmark = pd.DataFrame({
    "vendor_emp_ref": covered["bench_id"],
    "hris_link": covered["hris_id"],          # vendor was given the HRIS id to map
    "market_p50_salary": covered["market_benchmark_salary"],
    "market_p25_salary": (covered["market_benchmark_salary"] * 0.88).round(0),
    "market_p75_salary": (covered["market_benchmark_salary"] * 1.14).round(0),
    "benchmark_role": covered["job_level"],
})

# ---------------------------------------------------------------------------
# 6. Inject DELIBERATE data-quality problems for the ETL to catch & fix
# ---------------------------------------------------------------------------
# (a) Duplicate row in payroll (double export of one employee)
dup = payroll.iloc[[10]].copy()
payroll = pd.concat([payroll, dup], ignore_index=True)

# (b) A few missing salaries in payroll
miss_idx = RNG.choice(payroll.index[:N_EMPLOYEES], size=5, replace=False)
payroll.loc[miss_idx, "annual_base_eur"] = np.nan

# (c) Out-of-range / corrupt salary (data entry error: 10x)
payroll.loc[20, "annual_base_eur"] = payroll.loc[20, "annual_base_eur"] * 10

# (d) Missing performance ratings in HRIS
miss_perf = RNG.choice(hris.index, size=8, replace=False)
hris.loc[miss_perf, "Performance Rating"] = np.nan

# (e) Whitespace / casing inconsistencies in a join-key column (realistic dirty
#     text from a source export). The ETL normalises these with .str.strip() and
#     title-casing, so they should reconcile cleanly rather than fragment the data.
_dirty_idx = RNG.choice(hris.index, 6, replace=False)
_variants = {"Engineering": "  engineering ", "Sales": "SALES",
             "Support": "support ", "Marketing": " Marketing",
             "Finance": "finance", "Operations": "OPERATIONS "}
for _i in _dirty_idx:
    _d = hris.at[_i, "Department"]
    hris.at[_i, "Department"] = _variants.get(_d, _d)

# Shuffle row order in each source so they don't line up positionally
hris = hris.sample(frac=1, random_state=1).reset_index(drop=True)
payroll = payroll.sample(frac=1, random_state=2).reset_index(drop=True)
benchmark = benchmark.sample(frac=1, random_state=3).reset_index(drop=True)

hris.to_csv(RAW_DIR / "hris_dayforce.csv", index=False)
payroll.to_csv(RAW_DIR / "payroll_export.csv", index=False)
benchmark.to_csv(RAW_DIR / "market_benchmark.csv", index=False)

# Also stash a hidden ground-truth file (NOT used by ETL/model — for our own
# validation only) so we could audit leakage if needed.
df.drop(columns=["manager_idx"]).to_csv(
    RAW_DIR / "_ground_truth_DO_NOT_USE.csv", index=False)

print("\nWrote source files to data/raw/:")
for f in ["hris_dayforce.csv", "payroll_export.csv", "market_benchmark.csv"]:
    n = len(pd.read_csv(RAW_DIR / f))
    print(f"  {f:28s} {n:>4d} rows")
print("\nDeliberate data-quality issues injected: 1 duplicate, 5 missing salaries,")
print("1 out-of-range salary, 8 missing performance ratings, 12% benchmark gap.")
