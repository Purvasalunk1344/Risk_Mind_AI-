import joblib
import numpy as np
import pandas as pd

from config import MODELS


def _normalize_sex(value):
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().lower()
    mapping = {"m": "Male", "male": "Male", "man": "Male",
               "f": "Female", "female": "Female", "woman": "Female"}
    return mapping.get(text, str(value).title())


def _smoke_band(smoking):
    if smoking is None or pd.isna(smoking) or str(smoking).strip() == "":
        return "Unknown"
    smoking = float(smoking)
    if smoking == 0:
        return "None"
    if smoking < 15:
        return "Light"
    if smoking < 25:
        return "Moderate"
    return "Heavy"


def prepare_patient(raw_record):
    """Transform a raw patient dictionary to the feature layout expected by the model."""
    rec = dict(raw_record)

    # normalize input names
    if "Sex" in rec:
        rec["Sex"] = _normalize_sex(rec["Sex"])
    if "Smoking" in rec and "Smoke_Band" not in rec:
        rec["Smoke_Band"] = _smoke_band(rec["Smoking"])
    if "Smoke_Band" in rec and pd.isna(rec["Smoke_Band"]):
        rec["Smoke_Band"] = "Unknown"

    # BMI can be supplied directly or derived from weight and height
    if "BMI" not in rec or pd.isna(rec["BMI"]):
        weight = rec.get("Weight")
        height = rec.get("Height")
        if weight is not None and height is not None and height not in (0, None, ""):
            rec["BMI"] = float(weight) * 703.0 / float(height) ** 2
        else:
            rec["BMI"] = np.nan

    if "Cholesterol" in rec and pd.isna(rec["Cholesterol"]):
        rec["Chol_Missing"] = 1
    else:
        rec["Chol_Missing"] = 0

    if "Smoking" in rec and pd.isna(rec["Smoking"]):
        rec["Smoke_Missing"] = 1
    else:
        rec["Smoke_Missing"] = 0

    # if scorecard/monitoring record is already feature-ready, keep it as-is
    for c in ["AgeAtStart", "Systolic", "Diastolic", "Cholesterol", "BMI",
              "Chol_Missing", "Smoke_Missing", "Sex", "Smoke_Band"]:
        if c not in rec and c == "Smoke_Band":
            rec[c] = "Unknown"

    return rec


def score_patient(record, model_name="logistic", threshold=0.2, model_path=None):
    """Return a risk score and risk band for one patient record."""
    if model_path is None:
        model_path = MODELS / f"{model_name}.pkl"

    pipe, cols = joblib.load(model_path)
    patient = prepare_patient(record)
    X = pd.DataFrame([patient])
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"Missing model inputs: {missing}")
    prob = float(pipe.predict_proba(X[cols])[0, 1])
    band = "Low" if prob < 0.20 else "Medium" if prob < 0.50 else "High"
    return {
        "model": model_name,
        "risk": prob,
        "risk_band": band,
        "flagged": bool(prob >= threshold),
        "threshold": threshold,
    }
