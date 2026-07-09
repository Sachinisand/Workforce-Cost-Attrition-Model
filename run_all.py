"""
run_all.py — one-click pipeline runner
======================================
Runs the entire project end to end, in order, with clear progress markers.
Use this when you want to regenerate everything from scratch (e.g. live in an
interview) instead of running the five scripts one by one.

Usage, from the project root:
    python run_all.py

It will, in sequence:
    1. generate the synthetic source data
    2. integrate + govern it into one master dataset
    3. train the attrition model and score employees
    4. build the compensation cost scenario model (+ Excel)
    5. build the executive dashboard

At the end it prints where to find the dashboard so you can open it.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

STEPS = [
    ("01_generate_synthetic_data.py", "Generating synthetic source data"),
    ("02_etl_integration.py", "Integrating & governing data"),
    ("03_attrition_model.py", "Training attrition risk model"),
    ("04_compensation_scenario_model.py", "Building compensation scenario model"),
    ("05_build_dashboard.py", "Building executive dashboard"),
]

def main():
    print("=" * 70)
    print("WORKFORCE COST & ATTRITION RISK MODEL — FULL PIPELINE")
    print("=" * 70)
    for i, (script, label) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label} ...")
        print("-" * 70)
        result = subprocess.run([sys.executable, str(SRC / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"\n[ERROR] Step {i} ({script}) failed. Stopping.")
            sys.exit(1)
    dash = ROOT / "outputs" / "executive_dashboard.html"
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOpen the dashboard in your browser:\n  {dash}")
    print("\nKey outputs are in the outputs/ folder:")
    print("  - executive_dashboard.html        (open this to demo)")
    print("  - compensation_scenario_model.xlsx (the Excel model)")
    print("  - retention_watchlist.csv          (the actionable shortlist)")
    print("  - data_quality_report.txt          (ETL audit trail)")
    print("  - model_evaluation.txt             (model metrics, plain English)")

if __name__ == "__main__":
    main()
