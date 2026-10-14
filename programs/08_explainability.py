import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from config import ROOT, PROC, MODELS, PLOTS, SEED

REPORTS = ROOT / "output/reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# train = pd.read_csv(PROC / "train.csv")
# valid = pd.read_csv(PROC / "valid.csv")

train = pd.read_csv(PROC / "train.csv", keep_default_na=False, na_values=[""])
valid = pd.read_csv(PROC / "valid.csv", keep_default_na=False, na_values=[""])

NAMES = ["logistic", "random_forest", "gradient_boosting"]
loaded = {n: joblib.load(MODELS / f"{n}.pkl") for n in NAMES}


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


# ---------- 1. Odds ratios with confidence intervals (statsmodels) ----------
try:
    import statsmodels.formula.api as smf
    f = ("Dead ~ AgeAtStart + Systolic + Diastolic + Cholesterol + BMI "
         "+ C(Sex) + C(Smoke_Band, Treatment('None'))")
    sm = smf.logit(f, data=train).fit(disp=0)      # rows with missing values are dropped
    ci = np.exp(sm.conf_int())
    orr = pd.DataFrame({"odds_ratio": np.exp(sm.params), "ci_low": ci[0],
                        "ci_high": ci[1], "p_value": sm.pvalues}).drop("Intercept").round(3)
    print("Odds ratios (per unit for numeric variables):\n", orr)
    orr.to_csv(REPORTS / "odds_ratios.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(orr))
    ax.errorbar(orr["odds_ratio"], y,
                xerr=[orr["odds_ratio"] - orr["ci_low"], orr["ci_high"] - orr["odds_ratio"]],
                fmt="o", capsize=3)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_yticks(y, orr.index)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title("Logistic regression odds ratios")
    fig.savefig(PLOTS / "odds_ratios.png", dpi=120, bbox_inches="tight")
except Exception as e:
    print("Skipped odds ratios:", e, "\n(try: pip install --user statsmodels)")

# ---------- 2. Permutation importance (validation, AUC drop) ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, n in zip(axes, NAMES):
    pipe, cols = loaded[n]
    r = permutation_importance(pipe, valid[cols], valid["Dead"], scoring="roc_auc",
                               n_repeats=10, random_state=SEED, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=cols).sort_values()
    imp.plot.barh(ax=ax)
    ax.set_title(f"{n}: AUC drop when shuffled")
    print(f"\nPermutation importance - {n}:\n", imp.sort_values(ascending=False).round(4).head(6))
fig.savefig(PLOTS / "permutation_importance.png", dpi=120, bbox_inches="tight")

# ---------- 3. Partial dependence ----------
# fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# PartialDependenceDisplay.from_estimator(loaded["gradient_boosting"][0],
#                                         valid[loaded["gradient_boosting"][1]],
#                                         ["AgeAtStart", "Systolic", "Smoking"], ax=ax)
# fig.suptitle("Partial dependence - gradient boosting (probability of death)")
# fig.savefig(PLOTS / "partial_dependence.png", dpi=120, bbox_inches="tight")

# ---------- 3. Partial dependence ----------
gb_pipe, gb_cols = loaded["gradient_boosting"]
Xpd = valid[gb_cols].astype(
    {c: "float64" for c in valid[gb_cols].select_dtypes("number").columns})
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
PartialDependenceDisplay.from_estimator(gb_pipe, Xpd,
                                        ["AgeAtStart", "Systolic", "Smoking"], ax=ax)
fig.suptitle("Partial dependence - gradient boosting (probability of death)")
fig.savefig(PLOTS / "partial_dependence.png", dpi=120, bbox_inches="tight")

# ---------- 4. SHAP (optional) ----------
try:
    import shap
    pipe, cols = loaded["gradient_boosting"]
    pre = pipe.named_steps["pre"]
    X = dense(pre.transform(valid[cols]))
    names = [s.split("__", 1)[1] for s in pre.get_feature_names_out()]
    sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(X)
    plt.figure()
    shap.summary_plot(sv, X, feature_names=names, show=False)
    plt.savefig(PLOTS / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nSHAP summary saved.")
except Exception as e:
    print("\nSkipped SHAP:", e)

# ---------- 5. Plain-English explanation per patient (logistic model) ----------
pipe, cols = loaded["logistic"]
pre, lr = pipe.named_steps["pre"], pipe.named_steps["model"]
fnames = [s.split("__", 1)[1] for s in pre.get_feature_names_out()]
coefs = lr.coef_[0]
mu = dense(pre.transform(train[cols])).mean(axis=0)   # average patient
# LABEL = {"AgeAtStart": "age", "Systolic": "systolic BP", "Diastolic": "diastolic BP",
#          "Cholesterol": "cholesterol", "BMI": "BMI"}
LABEL = {"AgeAtStart": "age", "Systolic": "systolic BP"}


def describe(fn, row):
    if fn in LABEL:
        return f"{LABEL[fn]} {row[fn]:.0f}"
    if fn.startswith("Sex_") and row["Sex"] == fn.split("_", 1)[1]:
        return f"sex: {fn.split('_', 1)[1].lower()}"
    if fn.startswith("Smoke_Band_") and row["Smoke_Band"] == fn.split("_", 2)[2]:
        return f"smoking: {fn.split('_', 2)[2].lower()}"
    return None       # missing-value flags are not shown to the reader


def explain(row_df):
    p = pipe.predict_proba(row_df[cols])[0, 1]
    # contrib = dense(pre.transform(row_df[cols]))[0] * coefs
    contrib = (dense(pre.transform(row_df[cols]))[0] - mu) * coefs
    row = row_df.iloc[0]
    order = np.argsort(contrib)
    up = [describe(fnames[i], row) for i in order[::-1] if contrib[i] > 0]
    down = [describe(fnames[i], row) for i in order if contrib[i] < 0]
    up = [d for d in up if d][:3]
    down = [d for d in down if d][:2]
    text = f"Estimated mortality risk {p:.0%}."
    if up:
        text += " Factors raising risk: " + ", ".join(up) + "."
    if down:
        text += " Factors lowering risk: " + ", ".join(down) + "."
    return p, text


valid = valid.reset_index(drop=True)
valid["risk"] = pipe.predict_proba(valid[cols])[:, 1]
picks = [valid["risk"].idxmax(), valid["risk"].idxmin(),
         (valid["risk"] - valid["risk"].median()).abs().idxmin()]
picks += list(valid.sample(3, random_state=SEED).index)

out = []
for i in picks:
    p, text = explain(valid.loc[[i]])
    out.append({"Patient_ID": int(valid.loc[i, "Patient_ID"]), "risk": round(p, 3),
                "actual_dead": int(valid.loc[i, "Dead"]), "explanation": text})
    print(f"\nPatient {out[-1]['Patient_ID']} (actual outcome: {out[-1]['actual_dead']})\n  {text}")
pd.DataFrame(out).to_csv(REPORTS / "patient_explanations.csv", index=False)