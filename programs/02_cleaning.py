import numpy as np
import pandas as pd
from config import load_raw, PROC

df = load_raw()

df["Dead"] = (df["Status"] == "Dead").astype(int)
df["Chol_Missing"] = df["Cholesterol"].isna().astype(int)   # flags BEFORE imputing
df["Smoke_Missing"] = df["Smoking"].isna().astype(int)

for c in ["Smoking_Status", "BP_Status", "Chol_Status", "Weight_Status"]:
    df[c] = df[c].fillna("Unknown")

# Winsorize at 1st / 99th percentile
for c in ["Systolic", "Diastolic", "Cholesterol"]:
    lo, hi = df[c].quantile([0.01, 0.99])
    df[c] = df[c].clip(lo, hi)

# NOTE: missing-value imputation is done inside the model pipeline (05_modeling.py),
# fitted on the training data only, so no information leaks from validation/test.
df.to_csv(PROC / "heart_clean.csv", index=False)
print(df.shape, df.isna().sum()[lambda s: s > 0])