import numpy as np
import pandas as pd
from config import PROC

df = pd.read_csv(PROC / "heart_clean.csv")

df["BMI"] = df["Weight"] * 703 / df["Height"] ** 2            # inches, pounds
df["Pulse_Pressure"] = df["Systolic"] - df["Diastolic"]
df["MAP"] = df["Diastolic"] + (df["Systolic"] - df["Diastolic"]) / 3
df["Hypertension"] = ((df["Systolic"] >= 140) | (df["Diastolic"] >= 90)).astype(int)

df["Age_Group"] = pd.cut(df["AgeAtStart"], [0, 35, 45, 55, 200], right=False,
                         labels=["<35", "35-44", "45-54", "55+"]).astype(str)
df["Chol_Band"] = np.select([df["Cholesterol"] < 200, df["Cholesterol"] < 240,
                             df["Cholesterol"] >= 240],
                            ["Desirable", "Borderline", "High"], "Unknown")
df["Smoke_Band"] = np.select([df["Smoking"] == 0, df["Smoking"] < 15,
                              df["Smoking"] < 25, df["Smoking"] >= 25],
                             ["None", "Light", "Moderate", "Heavy"], "Unknown")

df["Age_x_Systolic"] = df["AgeAtStart"] * df["Systolic"]
df["Log_Chol"] = np.log(df["Cholesterol"])
df["Log_Smoking"] = np.log1p(df["Smoking"])

# survival frame keeps time-to-event columns (censoring logic comes in 07_survival.py)
df.to_csv(PROC / "heart_survival.csv", index=False)

# model frame: leakage columns removed
leak = ["AgeAtDeath", "DeathCause", "AgeCHDdiag", "Status"]
model_df = df.drop(columns=leak)
model_df.insert(0, "Patient_ID", range(1, len(model_df) + 1))
model_df.to_csv(PROC / "heart_model_ready.csv", index=False)
print(model_df.shape)