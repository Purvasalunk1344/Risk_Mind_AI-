"""
Shared serving and validation module for heart risk prediction models.
Used by backend (server.py), batch inference, and automated tests.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


VALIDATION_LIMITS = {
    "AgeAtStart": (18, 100, "Age must be between 18 and 100 years."),
    "Systolic": (70, 300, "Systolic BP must be between 70 and 300 mmHg."),
    "Diastolic": (40, 160, "Diastolic BP must be between 40 and 160 mmHg."),
    "Cholesterol": (80, 600, "Cholesterol must be between 80 and 600 mg/dL."),
    "Weight": (60, 500, "Weight must be between 60 and 500 lbs."),
    "Height": (48, 84, "Height must be between 48 and 84 inches."),
    "Smoking": (0, 80, "Smoking must be between 0 and 80 cigarettes/day."),
}


def normalize_sex(value: Any) -> str:
    """Normalize sex input to 'Male' or 'Female'."""
    if pd.isna(value) or value is None:
        return "Unknown"
    text = str(value).strip().lower()
    mapping = {
        "m": "Male", "male": "Male", "man": "Male",
        "f": "Female", "female": "Female", "woman": "Female"
    }
    return mapping.get(text, str(value).title())


def smoke_band(smoking: Any) -> str:
    """Map daily cigarettes to clinical smoking bands."""
    if smoking is None or pd.isna(smoking) or str(smoking).strip() == "":
        return "Unknown"
    try:
        smoking = float(smoking)
    except (ValueError, TypeError):
        return "Unknown"
    if smoking == 0:
        return "None"
    if smoking < 15:
        return "Light"
    if smoking < 25:
        return "Moderate"
    return "Heavy"


def validate_patient_input(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate patient inputs against physiological bounds."""
    for field, (low, high, msg) in VALIDATION_LIMITS.items():
        if field in record and record[field] is not None and record[field] != "":
            try:
                val = float(record[field])
                if val < low or val > high:
                    return False, f"Invalid {field} ({val}): {msg}"
            except (ValueError, TypeError):
                return False, f"Field '{field}' must be a valid number."
        else:
            return False, f"Missing required field: '{field}'."
    return True, None


def prepare_serving_patient(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw patient input dictionary into feature layout expected
    by all ML models (Logistic, Random Forest, Gradient Boosting, Cox).
    """
    rec = dict(raw_record)
    rec["Sex"] = normalize_sex(rec.get("Sex", "Female"))

    raw_age = rec.get("AgeAtStart")
    age = 50.0 if (raw_age is None or pd.isna(raw_age) or raw_age == "") else float(raw_age)

    raw_sys = rec.get("Systolic")
    systolic = 120.0 if (raw_sys is None or pd.isna(raw_sys) or raw_sys == "") else float(raw_sys)

    raw_dia = rec.get("Diastolic")
    diastolic = 80.0 if (raw_dia is None or pd.isna(raw_dia) or raw_dia == "") else float(raw_dia)

    raw_smk = rec.get("Smoking")
    smoking = 0.0 if (raw_smk is None or pd.isna(raw_smk) or raw_smk == "") else float(raw_smk)

    raw_wt = rec.get("Weight")
    weight = 160.0 if (raw_wt is None or pd.isna(raw_wt) or raw_wt == "") else float(raw_wt)

    raw_ht = rec.get("Height")
    height = 66.0 if (raw_ht is None or pd.isna(raw_ht) or raw_ht == "") else float(raw_ht)

    raw_chol = rec.get("Cholesterol")
    cholesterol = 200.0 if (raw_chol is None or pd.isna(raw_chol) or raw_chol == "") else float(raw_chol)

    rec["AgeAtStart"] = age
    rec["Systolic"] = systolic
    rec["Diastolic"] = diastolic
    rec["Smoking"] = smoking
    rec["Cholesterol"] = cholesterol

    bmi = (weight * 703.0 / (height ** 2)) if height > 0 else 25.0
    rec["BMI"] = bmi

    rec["Smoke_Band"] = smoke_band(smoking)
    rec["Chol_Missing"] = 1 if (raw_chol is None or pd.isna(raw_chol) or raw_chol == "") else 0
    rec["Smoke_Missing"] = 1 if (raw_smk is None or pd.isna(raw_smk) or raw_smk == "") else 0

    # Engineered clinical features for RF and Gradient Boosting
    rec["Pulse_Pressure"] = systolic - diastolic
    rec["MAP"] = diastolic + (systolic - diastolic) / 3.0
    rec["Hypertension"] = int(systolic >= 140 or diastolic >= 90)
    rec["Age_x_Systolic"] = age * systolic

    return rec
