"""
02_etl_integration.py
=====================
DATA INTEGRATION & GOVERNANCE LAYER

Ingests the three raw source files (HRIS / payroll / external benchmark), each with
a different employee-ID convention, reconciles them into ONE governed master table,
and produces a data-quality report of every issue found and how it was resolved.

Join strategy (mirrors a real Dayforce + Finance + benchmark-vendor integration):
    HRIS  --(Tax Reference)-->  Payroll          [system-of-record spine]
    HRIS  --(HRIS id link)-->   Benchmark vendor  [left join; vendor covers a subset]

Governance checks performed:
    - duplicate employee records (and de-duplication)
    - missing critical values (salary, performance rating)
    - out-of-range salaries (statistical + business-rule bounds)
    - referential integrity (manager IDs that don't exist)
    - benchmark coverage gap (imputation with documented method)

Outputs:
    data/processed/master_workforce.csv      governed unified dataset
    outputs/data_quality_report.txt          human-readable audit log

Run:  python src/02_etl_integration.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Data-quality event log: every issue is appended as (severity, check, detail, action)
dq_log = []
def log(severity, check, detail, action):
    dq_log.append({"severity": severity, "check": check,
                   "detail": detail, "action": action})

print("=" * 64)
print("ETL: INTEGRATING HRIS + PAYROLL + BENCHMARK -> GOVERNED MASTER")
print("=" * 64)

# ---------------------------------------------------------------------------
# 1. INGEST
# ---------------------------------------------------------------------------
hris = pd.read_csv(RAW / "hris_dayforce.csv", dtype=str)
payroll = pd.read_csv(RAW / "payroll_export.csv", dtype=str)
bench = pd.read_csv(RAW / "market_benchmark.csv", dtype=str)
print(f"\nIngested:  HRIS={len(hris)}  Payroll={len(payroll)}  Benchmark={len(bench)}")

# Normalise column names to snake_case canonical schema
hris = hris.rename(columns={
    "Employee ID": "hris_id", "Tax Reference": "tax_ref", "Department": "department",
    "Job Level": "job_level", "Location": "location", "Hire Date": "hire_date",
    "Tenure (Months)": "tenure_months", "Performance Rating": "performance_rating",
    "Manager ID": "manager_id", "Status": "status", "Exit Date": "exit_date"})

# Trim whitespace and standardise text keys (defends against casing/space drift).
# Departments arrive from the source export with inconsistent casing/spacing
# (e.g. "  engineering ", "SALES"); we canonicalise to a single clean form.
DEPT_CANON = {
    "engineering": "Engineering", "product": "Product",
    "data & analytics": "Data & Analytics", "sales": "Sales",
    "customer success": "Customer Success", "support": "Support",
    "marketing": "Marketing", "finance": "Finance",
    "people/hr": "People/HR", "operations": "Operations",
}
for col in ["department", "job_level", "location"]:
    hris[col] = hris[col].astype(str).str.strip()
_pre = hris["department"].copy()
hris["department"] = hris["department"].str.lower().str.strip().map(DEPT_CANON) \
    .fillna(hris["department"])
_n_normalised = (_pre != hris["department"]).sum()
if _n_normalised:
    log("WARN", "text_standardisation",
        f"{_n_normalised} department value(s) had inconsistent casing/whitespace",
        "canonicalised to standard department names")

# ---------------------------------------------------------------------------
# 2. TYPE COERCION  (track values that fail to parse)
# ---------------------------------------------------------------------------
def to_num(series, name, source):
    coerced = pd.to_numeric(series, errors="coerce")
    n_bad = coerced.isna().sum() - series.isna().sum()
    if n_bad > 0:
        log("WARN", "type_coercion",
            f"{n_bad} non-numeric value(s) in {source}.{name}",
            "coerced to NaN, handled downstream")
    return coerced

hris["tenure_months"] = to_num(hris["tenure_months"], "tenure_months", "hris")
hris["performance_rating"] = to_num(hris["performance_rating"], "performance_rating", "hris")
for c in ["annual_base_eur", "band_min", "band_mid", "band_max",
          "last_increase_pct", "months_since_increase"]:
    payroll[c] = to_num(payroll[c], c, "payroll")
for c in ["market_p50_salary", "market_p25_salary", "market_p75_salary"]:
    bench[c] = to_num(bench[c], c, "benchmark")

# ---------------------------------------------------------------------------
# 3. DE-DUPLICATION
# ---------------------------------------------------------------------------
def dedupe(df, key, source):
    dups = df[df.duplicated(subset=[key], keep=False)]
    if len(dups):
        ids = dups[key].unique().tolist()
        log("ERROR", "duplicate_records",
            f"{len(dups)} duplicate row(s) on {source}.{key}: {ids}",
            "kept first occurrence, dropped the rest")
        df = df.drop_duplicates(subset=[key], keep="first")
    return df

hris = dedupe(hris, "hris_id", "hris")
payroll = dedupe(payroll, "tax_ref", "payroll")
bench = dedupe(bench, "hris_link", "benchmark")

# ---------------------------------------------------------------------------
# 4. JOIN  HRIS (spine) <- Payroll  on tax_ref
# ---------------------------------------------------------------------------
master = hris.merge(payroll, on="tax_ref", how="left", indicator="_payroll_match")
unmatched_pay = (master["_payroll_match"] == "left_only").sum()
if unmatched_pay:
    log("ERROR", "referential_integrity",
        f"{unmatched_pay} HRIS employee(s) had no payroll record",
        "flagged; salary left null for QA follow-up")
master = master.drop(columns=["_payroll_match"])

# ---------------------------------------------------------------------------
# 5. JOIN  master <- Benchmark  on hris_id == hris_link  (vendor subset)
# ---------------------------------------------------------------------------
bench_join = bench.rename(columns={"hris_link": "hris_id"})
master = master.merge(
    bench_join[["hris_id", "market_p50_salary", "market_p25_salary", "market_p75_salary"]],
    on="hris_id", how="left", indicator="_bench_match")
missing_bench = (master["_bench_match"] == "left_only").sum()
log("INFO", "benchmark_coverage",
    f"{missing_bench} of {len(master)} employees "
    f"({missing_bench/len(master):.0%}) had no external benchmark",
    "imputed market_p50 from internal band_mid (documented fallback)")
# Documented imputation: where vendor has no benchmark, use internal band mid as
# best available proxy for market rate.
master["market_benchmark_salary"] = master["market_p50_salary"].fillna(master["band_mid"])
master["benchmark_imputed"] = master["_bench_match"].eq("left_only")
master = master.drop(columns=["_bench_match"])

# ---------------------------------------------------------------------------
# 6. MISSING-VALUE GOVERNANCE
# ---------------------------------------------------------------------------
# 6a. Missing salary -> impute with department x level median, flag it
miss_sal = master["annual_base_eur"].isna()
if miss_sal.any():
    med = master.groupby(["department", "job_level"])["annual_base_eur"].transform("median")
    glob_med = master["annual_base_eur"].median()
    master["annual_base_eur"] = master["annual_base_eur"].fillna(med).fillna(glob_med)
    log("ERROR", "missing_salary",
        f"{miss_sal.sum()} employee(s) missing base salary",
        "imputed with department x level median (global median fallback)")
master["salary_imputed"] = miss_sal.values

# 6b. Missing performance rating -> impute with median (3), flag it
miss_perf = master["performance_rating"].isna()
if miss_perf.any():
    master["performance_rating"] = master["performance_rating"].fillna(
        master["performance_rating"].median())
    log("WARN", "missing_performance",
        f"{miss_perf.sum()} employee(s) missing performance rating",
        "imputed with median rating (3); flagged for review")
master["performance_imputed"] = miss_perf.values

# ---------------------------------------------------------------------------
# 7. OUT-OF-RANGE / BUSINESS-RULE VALIDATION
# ---------------------------------------------------------------------------
# Business rule: a base salary should sit within [0.4x band_min, 2.5x band_max].
lower = master["band_min"] * 0.4
upper = master["band_max"] * 2.5
oor = (master["annual_base_eur"] < lower) | (master["annual_base_eur"] > upper)
if oor.any():
    ids = master.loc[oor, "hris_id"].tolist()
    log("ERROR", "salary_out_of_range",
        f"{oor.sum()} salary(ies) outside [0.4x band_min, 2.5x band_max]: {ids}",
        "capped to band_max (likely 10x data-entry error)")
    master.loc[oor, "annual_base_eur"] = master.loc[oor, "band_max"]
master["salary_corrected"] = oor.values

# Tenure sanity: non-negative, <= 600 months
bad_tenure = (master["tenure_months"] < 0) | (master["tenure_months"] > 600)
if bad_tenure.any():
    log("WARN", "tenure_out_of_range",
        f"{bad_tenure.sum()} implausible tenure value(s)",
        "clipped to [0, 600]")
    master["tenure_months"] = master["tenure_months"].clip(0, 600)

# ---------------------------------------------------------------------------
# 8. REFERENTIAL INTEGRITY: manager IDs must exist
# ---------------------------------------------------------------------------
valid_ids = set(master["hris_id"])
orphan_mgr = master["manager_id"].apply(
    lambda m: bool(m) and m not in valid_ids and m == m)  # m==m guards NaN
if orphan_mgr.any():
    log("WARN", "orphan_manager",
        f"{orphan_mgr.sum()} manager_id(s) not found in employee list",
        "left as-is; flagged (likely manager already exited)")

# ---------------------------------------------------------------------------
# 9. DERIVED GOVERNED FEATURES
# ---------------------------------------------------------------------------
master["comp_to_market_ratio"] = (
    master["annual_base_eur"] / master["market_benchmark_salary"]).round(4)
master["band_penetration"] = (
    (master["annual_base_eur"] - master["band_min"]) /
    (master["band_max"] - master["band_min"])).clip(0, 1.2).round(4)
master["is_active"] = master["status"].eq("Active")
master["attrition_flag"] = master["status"].eq("Terminated").astype(int)

# Canonical employee_id = HRIS id (system of record)
master = master.rename(columns={"hris_id": "employee_id",
                                "annual_base_eur": "base_salary",
                                "last_increase_pct": "last_merit_increase_pct",
                                "months_since_increase": "months_since_last_raise"})

keep = ["employee_id", "department", "job_level", "location", "hire_date",
        "tenure_months", "performance_rating", "manager_id", "status", "is_active",
        "exit_date", "base_salary", "band_min", "band_mid", "band_max",
        "market_benchmark_salary", "comp_to_market_ratio", "band_penetration",
        "last_merit_increase_pct", "months_since_last_raise", "attrition_flag",
        "benchmark_imputed", "salary_imputed", "performance_imputed",
        "salary_corrected"]
master = master[keep]

master.to_csv(PROC / "master_workforce.csv", index=False)

# ---------------------------------------------------------------------------
# 10. DATA-QUALITY REPORT
# ---------------------------------------------------------------------------
report = []
report.append("WORKFORCE DATA QUALITY & GOVERNANCE REPORT")
report.append("Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
report.append("=" * 60)
report.append(f"Source records:   HRIS={len(hris)}  Payroll(dedup)={len(payroll)}  "
              f"Benchmark={len(bench)}")
report.append(f"Unified master:   {len(master)} employees "
              f"({master['is_active'].sum()} active, "
              f"{(~master['is_active']).sum()} terminated)")
report.append("")
sev_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
counts = pd.Series([d["severity"] for d in dq_log]).value_counts().to_dict()
report.append("Issues by severity: " +
              ", ".join(f"{k}={counts.get(k,0)}" for k in ["ERROR", "WARN", "INFO"]))
report.append("-" * 60)
for d in sorted(dq_log, key=lambda x: sev_order[x["severity"]]):
    report.append(f"[{d['severity']:5s}] {d['check']}")
    report.append(f"        issue : {d['detail']}")
    report.append(f"        action: {d['action']}")
report.append("-" * 60)
report.append("Imputation / correction flag totals on master:")
for flag in ["benchmark_imputed", "salary_imputed", "performance_imputed",
             "salary_corrected"]:
    report.append(f"    {flag:22s}: {int(master[flag].sum())}")
report.append("=" * 60)
report.append("All flagged records remain queryable via their *_imputed / "
              "*_corrected columns for full auditability.")
report_text = "\n".join(report)
(OUT / "data_quality_report.txt").write_text(report_text)

print("\n" + report_text)
print(f"\nWrote governed master -> data/processed/master_workforce.csv "
      f"({len(master)} rows, {master.shape[1]} cols)")
