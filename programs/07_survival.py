import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import PROC, PLOTS

# df = pd.read_csv(PROC / "heart_survival.csv")
df = pd.read_csv(PROC / "heart_survival.csv", keep_default_na=False, na_values=[""])

# ---- Time and censoring ----
# SASHELP.HEART has no study-end date, so alive patients have no recorded follow-up length.
# ASSUMPTION: alive patients are censored after the longest follow-up seen among the dead.
df["Time"] = df["AgeAtDeath"] - df["AgeAtStart"]
MAX_FU = df.loc[df["Dead"] == 1, "Time"].max()
df.loc[df["Dead"] == 0, "Time"] = MAX_FU
print(f"Assumed censoring time for survivors: {MAX_FU:.0f} years")
print(df[["Time", "Dead"]].describe())

df["Male"] = (df["Sex"] == "Male").astype(int)
COVS = ["AgeAtStart", "Male", "Systolic", "Cholesterol", "BMI", "Smoking"]
sv = df[["Time", "Dead"] + COVS + ["Sex", "Smoke_Band", "BP_Status"]].copy()
for c in COVS:
    sv[c] = sv[c].fillna(sv[c].median())
sv = sv[sv["Time"] > 0]

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import multivariate_logrank_test
    LIB = "lifelines"
except Exception:
    LIB = "statsmodels"
print("survival library:", LIB)

GROUPS = ["Sex", "BP_Status", "Smoke_Band"]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

if LIB == "lifelines":
    for ax, g in zip(axes, GROUPS):
        for lvl, sub in sv.groupby(g):
            KaplanMeierFitter().fit(sub["Time"], sub["Dead"], label=str(lvl)).plot_survival_function(ax=ax, ci_show=False)
        p = multivariate_logrank_test(sv["Time"], sv[g], sv["Dead"]).p_value
        ax.set_title(f"Survival by {g} (log-rank p={p:.3g})")
        ax.set_xlabel("Years since baseline")
    cph = CoxPHFitter().fit(sv[["Time", "Dead"] + COVS], "Time", "Dead")
    cph.print_summary(decimals=3)
    print("\nHazard ratios:\n", np.exp(cph.params_).round(3))
    cph.check_assumptions(sv[["Time", "Dead"] + COVS], p_value_threshold=0.05, show_plots=False)
else:
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff
    from statsmodels.duration.hazard_regression import PHReg
    for ax, g in zip(axes, GROUPS):
        for lvl, sub in sv.groupby(g):
            sf = SurvfuncRight(sub["Time"], sub["Dead"], title=str(lvl))
            ax.step(sf.surv_times, sf.surv_prob, where="post", label=str(lvl))
        _, p = survdiff(sv["Time"], sv["Dead"], sv[g])
        ax.set_title(f"Survival by {g} (log-rank p={p:.3g})")
        ax.set_xlabel("Years since baseline"); ax.legend()
    res = PHReg(sv["Time"], sv[COVS], status=sv["Dead"]).fit()
    print(res.summary())
    print("\nHazard ratios:\n", pd.Series(np.exp(res.params), index=COVS).round(3))

for ax in axes:
    ax.set_ylabel("Survival probability")
fig.savefig(PLOTS / "km_curves.png", dpi=120, bbox_inches="tight")