import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import ROOT, PROC, MODELS, PLOTS

REPORTS = ROOT / "output/reports"
REPORTS.mkdir(parents=True, exist_ok=True)
rd = dict(keep_default_na=False, na_values=[""])
train = pd.read_csv(PROC / "train.csv", **rd)
valid = pd.read_csv(PROC / "valid.csv", **rd)
test = pd.read_csv(PROC / "test.csv", **rd)

pipe, cols = joblib.load(MODELS / "logistic.pkl")
for d in (train, valid, test):
    d["risk"] = pipe.predict_proba(d[cols])[:, 1]

# ---------- 1. Risk bands from the champion model ----------
# Cutoffs: 0.20 = the 3:1 cost threshold, 0.50 = roughly the 1:1 threshold (see 06).
LOW, HIGH = 0.20, 0.50


def band(p):
    return pd.cut(p, [-1, LOW, HIGH, 2], labels=["Low", "Medium", "High"], right=False)


def band_table(d):
    g = d.assign(band=band(d["risk"])).groupby("band", observed=True)
    return g.agg(patients=("Dead", "size"), deaths=("Dead", "sum"),
                 observed_death_rate=("Dead", "mean"),
                 mean_predicted=("risk", "mean")).round(3)


bt_valid, bt_test = band_table(valid), band_table(test)
print("Risk bands - validation:\n", bt_valid)
print("\nRisk bands - test:\n", bt_test)
pd.concat({"valid": bt_valid, "test": bt_test}).to_csv(REPORTS / "risk_bands.csv")

fig, ax = plt.subplots(figsize=(6, 4))
w = 0.38
x = np.arange(len(bt_valid))
ax.bar(x - w / 2, bt_valid["observed_death_rate"], w, label="validation")
ax.bar(x + w / 2, bt_test["observed_death_rate"].reindex(bt_valid.index), w, label="test")
ax.set_xticks(x, bt_valid.index)
ax.set_ylabel("Observed death rate")
ax.set_title("Observed death rate by risk band")
ax.legend()
fig.savefig(PLOTS / "risk_bands.png", dpi=120, bbox_inches="tight")

# ---------- 2. Fairness: performance by subgroup ----------
# Validation + test are combined for larger groups. This is reporting only, no tuning.
comb = pd.concat([valid, test], ignore_index=True)
comb["Age_Group"] = comb["Age_Group"].astype(str)


def subgroup(d, col):
    rows = []
    for lvl, s in d.groupby(col):
        both = s["Dead"].nunique() == 2
        flagged = s["risk"] >= LOW
        deaths = int(s["Dead"].sum())
        rows.append({"group": f"{col}={lvl}", "n": len(s),
                     "death_rate": s["Dead"].mean(),
                     "mean_predicted": s["risk"].mean(),
                     "auc": roc_auc_score(s["Dead"], s["risk"]) if both else np.nan,
                     "recall@0.20": (flagged & (s["Dead"] == 1)).sum() / max(deaths, 1),
                     "share_flagged": flagged.mean()})
    return pd.DataFrame(rows)


fair = pd.concat([subgroup(comb, "Sex"), subgroup(comb, "Age_Group")]).round(3)
print("\nFairness by subgroup (validation + test):\n", fair.to_string(index=False))
fair.to_csv(REPORTS / "fairness_by_group.csv", index=False)

# ---------- 3. Points-based scorecard ----------
SC_CATS = {"Age_Group": "<35", "Sex": "Female", "Smoke_Band": "None", "BP_Status": "Optimal"}
for d in (train, valid, test):
    d["Age_Group"] = d["Age_Group"].astype(str)


def design(d):
    X = pd.DataFrame(index=d.index)
    for c, ref in SC_CATS.items():
        for lvl in sorted(train[c].unique()):
            if lvl != ref and lvl != "Unknown":
                X[f"{c}={lvl}"] = (d[c] == lvl).astype(int)
    return X


fit_rows = train["Smoke_Band"] != "Unknown"
Xtr = design(train)
lr = LogisticRegression(C=1e6, max_iter=1000).fit(Xtr[fit_rows], train.loc[fit_rows, "Dead"])

PDO = 20                                  # 20 points = odds of death double
factor = PDO / np.log(2)
points = pd.Series((lr.coef_[0] * factor).round().astype(int), index=Xtr.columns)
print(f"\nScorecard points ({PDO} points = odds double; reference levels = 0 points):\n",
      points.to_string())
points.rename("points").to_csv(REPORTS / "scorecard_points.csv")


def score(d):
    return design(d)[points.index].values @ points.values


s_tr, s_va, s_te = score(train), score(valid), score(test)
print(f"\nScorecard AUC - validation: {roc_auc_score(valid['Dead'], s_va):.3f}  "
      f"test: {roc_auc_score(test['Dead'], s_te):.3f}")

q1, q2 = np.quantile(s_tr, [0.33, 0.66])      # band cutoffs fixed from the training set
labels = ["Low", "Medium", "High"]
for name, d, s in [("validation", valid, s_va), ("test", test, s_te)]:
    b = pd.cut(s, [-np.inf, q1, q2, np.inf], labels=labels)
    t = d.assign(score_band=b).groupby("score_band", observed=True).agg(
        patients=("Dead", "size"), observed_death_rate=("Dead", "mean")).round(3)
    print(f"\nScore bands (cutoffs {q1:.0f} / {q2:.0f}) - {name}:\n", t)