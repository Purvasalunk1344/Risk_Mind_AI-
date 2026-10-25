"""
12_survival_model.py: Train and persist Cox Proportional Hazards survival model.
Uses identical time-to-event censoring and covariates as 07_survival.py.
Persists model to models/survival_cox.pkl for serving in backend.
"""

import joblib
import pandas as pd
from lifelines import CoxPHFitter

from config import PROC, MODELS


def train_survival_model():
    print("[*] Loading heart_survival.csv...")
    df = pd.read_csv(PROC / "heart_survival.csv", keep_default_na=False, na_values=[""])

    # ---- Time and censoring ----
    # SASHELP.HEART has no study-end date, so alive patients have no recorded follow-up length.
    # ASSUMPTION: alive patients are censored after the longest follow-up seen among the dead.
    df["Time"] = df["AgeAtDeath"] - df["AgeAtStart"]
    MAX_FU = float(df.loc[df["Dead"] == 1, "Time"].max())
    df.loc[df["Dead"] == 0, "Time"] = MAX_FU
    print(f"[*] Assumed censoring time for survivors: {MAX_FU:.1f} years")

    df["Male"] = (df["Sex"] == "Male").astype(int)
    COVS = ["AgeAtStart", "Male", "Systolic", "Cholesterol", "BMI", "Smoking"]

    sv = df[["Time", "Dead"] + COVS].copy()
    for c in COVS:
        sv[c] = sv[c].fillna(sv[c].median())
    sv = sv[sv["Time"] > 0]

    print(f"[*] Fitting CoxPHFitter on {len(sv)} valid patient records...")
    cph = CoxPHFitter()
    cph.fit(sv[["Time", "Dead"] + COVS], duration_col="Time", event_col="Dead")

    concordance = cph.concordance_index_
    print(f"[*] Cox Model Concordance Index (C-index): {concordance:.4f}")
    print("\nModel Summary:")
    cph.print_summary(decimals=3)

    out_path = MODELS / "survival_cox.pkl"
    joblib.dump((cph, COVS, MAX_FU), out_path)
    print(f"[OK] Persisted survival Cox model and metadata to {out_path}")
    return cph, concordance


if __name__ == "__main__":
    train_survival_model()
