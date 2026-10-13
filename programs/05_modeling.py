import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import PROC, MODELS, SEED
from macros.metrics import evaluate
from macros.log_experiment import log_experiment

train = pd.read_csv(PROC / "train.csv")
valid = pd.read_csv(PROC / "valid.csv")

CAT = ["Sex", "Smoke_Band"]
NUM_LOGIT = ["AgeAtStart", "Systolic", "Diastolic", "Cholesterol", "BMI",
             "Chol_Missing", "Smoke_Missing"]          # no collinear duplicates
NUM_TREE = NUM_LOGIT + ["Smoking", "Pulse_Pressure", "MAP", "Age_x_Systolic", "Hypertension"]


def make_pipeline(num, model, scale=False):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "logistic": (NUM_LOGIT, True,
                 LogisticRegression(max_iter=1000, random_state=SEED)),
    "random_forest": (NUM_TREE, False,
                      RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                             n_jobs=2, random_state=SEED)),
    "gradient_boosting": (NUM_TREE, False,
                          GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                     max_depth=3, random_state=SEED)),
}

results = {}
for name, (num, scale, est) in models.items():
    pipe = make_pipeline(num, est, scale)
    pipe.fit(train[num + CAT], train["Dead"])
    p = pipe.predict_proba(valid[num + CAT])[:, 1]
    m = evaluate(valid["Dead"], p)
    results[name] = m
    log_experiment(name, est.get_params(), m)
    joblib.dump((pipe, num + CAT), MODELS / f"{name}.pkl")

summary = pd.DataFrame(results).T.sort_values("auc", ascending=False).round(4)
print(summary)
print("champion:", summary.index[0])