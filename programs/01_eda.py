import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import load_raw, RAW, PLOTS

df = load_raw()
df.to_csv(RAW / "heart_raw.csv", index=False)

print(df.shape)
print(df.dtypes)
print(df["Status"].value_counts(normalize=True).round(4))
print(df.isna().sum())
print(df.describe().T)
print("duplicate rows:", df.duplicated().sum())

for c in ["Sex", "Smoking_Status", "BP_Status", "Chol_Status", "Weight_Status"]:
    print(pd.crosstab(df[c], df["Status"], normalize="index").round(3))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["AgeAtStart", "Systolic", "Cholesterol"]):
    for s, g in df.groupby("Status"):
        a.hist(g[col].dropna(), bins=30, alpha=0.5, label=s)
    a.set_title(col)
    a.legend()
fig.savefig(PLOTS / "eda_distributions.png", dpi=120, bbox_inches="tight")

num = ["AgeAtStart", "Systolic", "Diastolic", "Cholesterol", "Weight", "Smoking"]
corr = df[num].corr()
fig, a = plt.subplots(figsize=(6, 5))
im = a.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
a.set_xticks(range(len(num)), num, rotation=45, ha="right")
a.set_yticks(range(len(num)), num)
fig.colorbar(im)
fig.savefig(PLOTS / "eda_corr.png", dpi=120, bbox_inches="tight")