"""
Step 2 Consistency Check:
Compare Logistic 32-year mortality risk with 1 - S(32 years) from CoxPHFitter
for 20 random patients sampled from test.csv.
Computes Pearson correlation and Mean Absolute Gap (MAE).
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "programs"))

from config import PROC, MODELS, SEED
from macros.serving import prepare_serving_patient


def run_consistency_check():
    # Load logistic pipeline
    log_pipe, log_cols = joblib.load(MODELS / "logistic.pkl")

    # Load Cox model
    cph, cox_cols, max_fu = joblib.load(MODELS / "survival_cox.pkl")

    # Load test dataset
    test_df = pd.read_csv(PROC / "test.csv", keep_default_na=False, na_values=[""])

    # Sample 20 random patients
    sample = test_df.sample(20, random_state=SEED).copy()

    logistic_risks = []
    cox_32yr_risks = []
    patient_summaries = []

    for idx, row in sample.iterrows():
        raw = row.to_dict()
        prep = prepare_serving_patient(raw)

        # 1. Logistic probability
        df_log = pd.DataFrame([prep])
        log_prob = float(log_pipe.predict_proba(df_log[log_cols])[0, 1])
        logistic_risks.append(log_prob)

        # 2. Cox probability: 1 - S(32)
        # Prepare row with exact columns for Cox: AgeAtStart, Male, Systolic, Cholesterol, BMI, Smoking
        male_val = 1 if prep["Sex"] == "Male" else 0
        cox_row = pd.DataFrame([{
            "AgeAtStart": prep["AgeAtStart"],
            "Male": male_val,
            "Systolic": prep["Systolic"],
            "Cholesterol": prep["Cholesterol"],
            "BMI": prep["BMI"],
            "Smoking": prep["Smoking"]
        }])

        surv_fn = cph.predict_survival_function(cox_row, times=[32.0])
        s_32 = float(surv_fn.iloc[0, 0])
        cox_risk = 1.0 - s_32
        cox_32yr_risks.append(cox_risk)

        patient_summaries.append({
            "Age": int(prep["AgeAtStart"]),
            "Sex": prep["Sex"],
            "BP": f"{int(prep['Systolic'])}/{int(prep['Diastolic'])}",
            "Cigs": int(prep["Smoking"]),
            "Logit_Risk": log_prob,
            "Cox_32yr_Risk": cox_risk,
            "Abs_Gap": abs(log_prob - cox_risk)
        })

    log_arr = np.array(logistic_risks)
    cox_arr = np.array(cox_32yr_risks)

    correlation = float(np.corrcoef(log_arr, cox_arr)[0, 1])
    mean_abs_gap = float(np.mean(np.abs(log_arr - cox_arr)))
    max_gap = float(np.max(np.abs(log_arr - cox_arr)))

    print("=" * 80)
    print("STEP 2: LOGISTIC RISK vs. COX 1 - S(32 years) CONSISTENCY CHECK (20 PATIENTS)")
    print("=" * 80)
    print(f"{'#':<3} | {'Demographics':<16} | {'BP (mmHg)':<10} | {'Cigs':<5} | {'Logistic':<10} | {'Cox 1-S(32)':<12} | {'Abs Gap':<8}")
    print("-" * 80)
    for i, p in enumerate(patient_summaries, 1):
        demo = f"{p['Age']}yo {p['Sex']}"
        print(f"{i:<3} | {demo:<16} | {p['BP']:<10} | {p['Cigs']:<5} | {p['Logit_Risk']:<10.4f} | {p['Cox_32yr_Risk']:<12.4f} | {p['Abs_Gap']:<8.4f}")

    print("=" * 80)
    print(f"Pearson Correlation (r) : {correlation:.4f}")
    print(f"Mean Absolute Gap (MAE) : {mean_abs_gap:.4f}")
    print(f"Max Absolute Gap        : {max_gap:.4f}")
    print("=" * 80)

    return correlation, mean_abs_gap


if __name__ == "__main__":
    run_consistency_check()
