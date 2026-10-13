import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, confusion_matrix

from config import PROC, MODELS, MON, PLOTS
from macros.metrics import evaluate

# valid = pd.read_csv(PROC / "valid.csv")
# test = pd.read_csv(PROC / "test.csv")

valid = pd.read_csv(PROC / "valid.csv", keep_default_na=False, na_values=[""])
test = pd.read_csv(PROC / "test.csv", keep_default_na=False, na_values=[""])
NAMES = ["logistic", "random_forest", "gradient_boosting"]

loaded = {n: joblib.load(MODELS / f"{n}.pkl") for n in NAMES}


def predict(name, df):
    pipe, cols = loaded[name]
    return pipe.predict_proba(df[cols])[:, 1]


vp = {n: predict(n, valid) for n in NAMES}

# 1. ROC curves (validation)
fig, ax = plt.subplots(figsize=(6, 5))
for n in NAMES:
    fpr, tpr, _ = roc_curve(valid["Dead"], vp[n])
    ax.plot(fpr, tpr, label=f"{n} (AUC={evaluate(valid['Dead'], vp[n])['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curves - validation"); ax.legend()
fig.savefig(PLOTS / "roc_valid.png", dpi=120, bbox_inches="tight")

# 2. Calibration curves (validation)
fig, ax = plt.subplots(figsize=(6, 5))
for n in NAMES:
    frac, mean_p = calibration_curve(valid["Dead"], vp[n], n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, marker="o", label=n)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed death rate")
ax.set_title("Calibration - validation"); ax.legend()
fig.savefig(PLOTS / "calibration_valid.png", dpi=120, bbox_inches="tight")

# 3. Champion = best validation AUC
champ = max(NAMES, key=lambda n: evaluate(valid["Dead"], vp[n])["auc"])
print("champion:", champ)

# 4. Cost-based threshold, chosen on VALIDATION only.
#    ASSUMPTION: missing a death costs 5x a false alarm. Change COST_FN to test other values.
COST_FN, COST_FP = 5, 1
grid = np.linspace(0.05, 0.95, 91)
costs = []
for t in grid:
    tn, fp, fn, tp = confusion_matrix(valid["Dead"], (vp[champ] >= t).astype(int)).ravel()
    costs.append(COST_FN * fn + COST_FP * fp)
best_t = float(grid[int(np.argmin(costs))])
print(f"best threshold (FN cost {COST_FN}x): {best_t:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(grid, costs); ax.axvline(best_t, color="r", ls="--")
ax.set_xlabel("Threshold"); ax.set_ylabel("Total cost"); ax.set_title("Cost vs threshold")
fig.savefig(PLOTS / "threshold_cost.png", dpi=120, bbox_inches="tight")

print("validation @0.5      :", {k: round(v, 3) for k, v in evaluate(valid["Dead"], vp[champ], 0.5).items()})
print("validation @best_t   :", {k: round(v, 3) for k, v in evaluate(valid["Dead"], vp[champ], best_t).items()})

# 5. FINAL test-set check: champion only, threshold fixed from validation
tp_ = predict(champ, test)
res = evaluate(test["Dead"], tp_, best_t)
print("TEST @best_t         :", {k: round(v, 3) for k, v in res.items()})
print("TEST confusion matrix [[TN FP] [FN TP]]:\n",
      confusion_matrix(test["Dead"], (tp_ >= best_t).astype(int)))

json.dump({"champion": champ, "threshold": best_t, "cost_fn": COST_FN,
           "test_metrics": res}, open(MON / "champion.json", "w"), indent=2)

# 6. Cost sensitivity table (validation only)
rows = []
for fn_cost in [1, 2, 3, 5, 10]:
    cs = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(valid["Dead"], (vp[champ] >= t).astype(int)).ravel()
        cs.append(fn_cost * fn + fp)
    t_best = float(grid[int(np.argmin(cs))])
    m = evaluate(valid["Dead"], vp[champ], t_best)
    flagged = float((vp[champ] >= t_best).mean())
    rows.append({"FN:FP cost": f"{fn_cost}:1", "threshold": round(t_best, 2),
                 "recall": round(m["recall"], 3), "precision": round(m["precision"], 3),
                 "share_flagged": round(flagged, 3)})
sens = pd.DataFrame(rows)
print("\nCost sensitivity (validation):\n", sens.to_string(index=False))
sens.to_csv(MON / "threshold_sensitivity.csv", index=False)