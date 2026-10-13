import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROC, SEED

df = pd.read_csv(PROC / "heart_model_ready.csv")

train, rest = train_test_split(df, test_size=0.4, stratify=df["Dead"], random_state=SEED)
valid, test = train_test_split(rest, test_size=0.5, stratify=rest["Dead"], random_state=SEED + 1)

for name, d in [("train", train), ("valid", valid), ("test", test)]:
    d.to_csv(PROC / f"{name}.csv", index=False)
    print(name, d.shape, "dead rate:", round(d["Dead"].mean(), 4))