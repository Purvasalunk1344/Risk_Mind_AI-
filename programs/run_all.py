"""
End-to-end pipeline runner for Heart Disease Mortality Risk System.
Executes scripts 01 through 10 in sequential order.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    ("01_eda.py", "Exploratory Data Analysis & Visualizations"),
    ("02_cleaning.py", "Data Cleaning & Preprocessing"),
    ("03_feature_engineering.py", "Clinical Feature Engineering"),
    ("04_split.py", "Train / Validation / Test Stratified Split"),
    ("05_modeling.py", "Model Training (Logistic, RF, Gradient Boosting)"),
    ("06_evaluation.py", "Model Evaluation & Calibration"),
    ("07_survival.py", "Survival Analysis (Kaplan-Meier & Cox PH)"),
    ("08_explainability.py", "Model Explainability & Odds Ratios"),
    ("09_scorecard_fairness.py", "Scorecard & Subgroup Fairness"),
    ("10_mlops_monitoring.py", "MLOps Drift Monitoring & PSI"),
    ("12_survival_model.py", "Cox Proportional Hazards Model Training & Export"),
]

def run_pipeline():
    programs_dir = Path(__file__).resolve().parent
    print("=" * 70)
    print(" HEART RISK MORTALITY PIPELINE - END-TO-END RUNNER")
    print("=" * 70)

    total_start = time.time()
    for script_name, description in SCRIPTS:
        script_path = programs_dir / script_name
        if not script_path.exists():
            print(f"[!] Warning: {script_name} not found, skipping.")
            continue

        print(f"\n--> Running [{script_name}]: {description} ...")
        t0 = time.time()
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(programs_dir))
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"[ERROR] Step {script_name} failed with return code {res.returncode}.")
            sys.exit(res.returncode)
        print(f"[OK] Completed {script_name} in {elapsed:.1f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"[SUCCESS] All pipeline steps completed in {total_time:.1f} seconds!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
