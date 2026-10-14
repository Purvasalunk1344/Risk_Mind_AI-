import json
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import MON, MODELS, PLOTS, PROC
from macros.metrics import evaluate
from macros.psi import calculate_psi


MODEL_NAME = "logistic"
TRAIN = pd.read_csv(PROC / "train.csv", keep_default_na=False, na_values=[""])
VALID = pd.read_csv(PROC / "valid.csv", keep_default_na=False, na_values=[""])
TEST = pd.read_csv(PROC / "test.csv", keep_default_na=False, na_values=[""])
MODEL, COLS = joblib.load(MODELS / f"{MODEL_NAME}.pkl")

for d in (TRAIN, VALID, TEST):
    d["risk"] = MODEL.predict_proba(d[COLS])[:, 1]


def build_summary():
    valid_auc = evaluate(VALID["Dead"], VALID["risk"])['auc']
    test_auc = evaluate(TEST["Dead"], TEST["risk"])['auc']
    valid_psi = calculate_psi(TRAIN["risk"], VALID["risk"])
    test_psi = calculate_psi(TRAIN["risk"], TEST["risk"])

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "validation_auc": round(float(valid_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "validation_psi": round(float(valid_psi["psi"]), 4),
        "test_psi": round(float(test_psi["psi"]), 4),
        "risk_threshold": 0.20,
        "risk_threshold_high": 0.50,
    }
    return summary


summary = build_summary()
print(json.dumps(summary, indent=2))

# Drift monitoring: PSI between train and later slices
for name, df in [("valid", VALID), ("test", TEST)]:
    psi = calculate_psi(TRAIN["risk"], df["risk"], bins=10)
    psi["breakdown"].to_csv(MON / f"psi_{name}.csv", index=False)

# Monitor model performance and save summary to monitoring folder
with open(MON / "monitoring_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# score distribution plot to support drift monitoring
fig, ax = plt.subplots(figsize=(6, 4))
for label, df in [("train", TRAIN), ("validation", VALID), ("test", TEST)]:
    ax.hist(df["risk"], bins=20, alpha=0.5, label=label)
ax.set_xlabel("Predicted risk")
ax.set_ylabel("Count")
ax.set_title("Risk-score distribution by split")
ax.legend()
fig.tight_layout()
fig.savefig(PLOTS / "risk_distribution_monitoring.png", dpi=120, bbox_inches="tight")

# Save a tabular monitor snapshot
monitor_tbl = pd.DataFrame([
    {
        "dataset": "validation",
        "n": len(VALID),
        "dead_rate": round(float(VALID["Dead"].mean()), 4),
        "auc": round(float(evaluate(VALID["Dead"], VALID["risk"])["auc"]), 4),
        "psi_vs_train": round(float(calculate_psi(TRAIN["risk"], VALID["risk"])["psi"]), 4),
    },
    {
        "dataset": "test",
        "n": len(TEST),
        "dead_rate": round(float(TEST["Dead"].mean()), 4),
        "auc": round(float(evaluate(TEST["Dead"], TEST["risk"])["auc"]), 4),
        "psi_vs_train": round(float(calculate_psi(TRAIN["risk"], TEST["risk"])["psi"]), 4),
    },
])
monitor_tbl.to_csv(MON / "model_monitoring.csv", index=False)

print("\nMonitoring summary saved to:", MON / "monitoring_summary.json")
print("Drift tables saved to:", MON / "psi_valid.csv", MON / "psi_test.csv")
