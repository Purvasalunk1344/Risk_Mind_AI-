"""
Comprehensive test suite for heart risk prediction system:
1. /predict equals pipe.predict_proba for 3 fixed patients.
2. Invalid input returns an error (bounds checking).
3. Survival values never increase with time (monotonic non-increasing survival).
4. The controlled scenario goes through the model.
5. scoreLogic.js contains no calculateRisk.
"""

import sys
from pathlib import Path
import json
import pytest
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "programs"))

from backend.server import app
from programs.macros.serving import prepare_serving_patient, VALIDATION_LIMITS


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


FIXED_PATIENTS = [
    {
        "name": "Fixed Patient 1: 58F, 145/88, Chol 210, 160lb, 66in, 12 cigs",
        "data": {
            "AgeAtStart": 58,
            "Sex": "Female",
            "Systolic": 145,
            "Diastolic": 88,
            "Cholesterol": 210,
            "Weight": 160,
            "Height": 66,
            "Smoking": 12,
        }
    },
    {
        "name": "Fixed Patient 2: 25F, 110/70, Chol 160, 120lb, 64in, 0 cigs",
        "data": {
            "AgeAtStart": 25,
            "Sex": "Female",
            "Systolic": 110,
            "Diastolic": 70,
            "Cholesterol": 160,
            "Weight": 120,
            "Height": 64,
            "Smoking": 0,
        }
    },
    {
        "name": "Fixed Patient 3: 65M, 165/95, Chol 260, 205lb, 70in, 30 cigs",
        "data": {
            "AgeAtStart": 65,
            "Sex": "Male",
            "Systolic": 165,
            "Diastolic": 95,
            "Cholesterol": 260,
            "Weight": 205,
            "Height": 70,
            "Smoking": 30,
        }
    },
]


# ==============================================================================
# TEST 1: /predict equals pipe.predict_proba for 3 fixed patients
# ==============================================================================
@pytest.mark.parametrize("patient_fixture", FIXED_PATIENTS)
def test_predict_equals_pipeline(client, patient_fixture):
    pipe, cols = joblib.load(ROOT / "models" / "logistic.pkl")
    raw = patient_fixture["data"]

    # Direct model prediction
    prep = prepare_serving_patient(raw)
    df = pd.DataFrame([prep])
    direct_prob = float(pipe.predict_proba(df[cols])[0, 1])

    # API /predict response
    res = client.post("/predict", json=raw)
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True

    # Must match to 4 decimal places
    assert round(data["risk"], 4) == round(direct_prob, 4)


# ==============================================================================
# TEST 2: Invalid input returns an error (bounds checking)
# age 18-100, systolic 70-300, diastolic 40-160, cholesterol 80-600, weight 60-500, height 48-84, smoking 0-80
# ==============================================================================
INVALID_CASES = [
    ("AgeAtStart", 17, "Age too low"),
    ("AgeAtStart", 101, "Age too high"),
    ("Systolic", 69, "Systolic too low"),
    ("Systolic", 301, "Systolic too high"),
    ("Diastolic", 39, "Diastolic too low"),
    ("Diastolic", 161, "Diastolic too high"),
    ("Cholesterol", 79, "Cholesterol too low"),
    ("Cholesterol", 601, "Cholesterol too high"),
    ("Weight", 59, "Weight too low"),
    ("Weight", 501, "Weight too high"),
    ("Height", 47, "Height too low"),
    ("Height", 85, "Height too high"),
    ("Smoking", -1, "Smoking negative"),
    ("Smoking", 81, "Smoking too high"),
]


@pytest.mark.parametrize("field, bad_val, case_desc", INVALID_CASES)
def test_invalid_input_returns_error(client, field, bad_val, case_desc):
    base = {
        "AgeAtStart": 50,
        "Sex": "Female",
        "Systolic": 120,
        "Diastolic": 80,
        "Cholesterol": 200,
        "Weight": 150,
        "Height": 65,
        "Smoking": 0,
    }
    base[field] = bad_val
    res = client.post("/predict", json=base)
    assert res.status_code == 400, f"Expected 400 for {case_desc}, got {res.status_code}"
    data = res.get_json()
    assert data["success"] is False
    assert "error" in data


# ==============================================================================
# TEST 3: Survival values never increase with time
# S(t2) <= S(t1) for t2 > t1
# ==============================================================================
def test_survival_values_never_increase(client):
    for patient_fixture in FIXED_PATIENTS:
        raw = patient_fixture["data"]
        res = client.post("/survival", json=raw)
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True

        trajectory = data["trajectory"]
        assert len(trajectory) >= 6

        # Untreated survival non-increasing
        for i in range(1, len(trajectory)):
            prev_s = trajectory[i - 1]["untreatedSurvival"]
            curr_s = trajectory[i]["untreatedSurvival"]
            assert curr_s <= prev_s + 1e-5, (
                f"Survival increased at year {trajectory[i]['year']}: {prev_s} -> {curr_s}"
            )

        # Controlled survival non-increasing
        for i in range(1, len(trajectory)):
            prev_s = trajectory[i - 1]["controlledSurvival"]
            curr_s = trajectory[i]["controlledSurvival"]
            assert curr_s <= prev_s + 1e-5, (
                f"Controlled survival increased at year {trajectory[i]['year']}: {prev_s} -> {curr_s}"
            )


# ==============================================================================
# TEST 4: The controlled scenario goes through the model
# ==============================================================================
def test_controlled_scenario_goes_through_model(client):
    pipe, cols = joblib.load(ROOT / "models" / "logistic.pkl")

    for patient_fixture in FIXED_PATIENTS:
        raw = patient_fixture["data"]

        # Call API /predict
        res = client.post("/predict", json=raw)
        assert res.status_code == 200
        data = res.get_json()
        api_controlled_risk = data["controlled_risk"]

        # Directly prepare controlled record: Systolic <= 120, Diastolic <= 80, Smoking = 0
        controlled_raw = dict(raw)
        controlled_raw["Systolic"] = min(float(raw["Systolic"]), 120.0)
        controlled_raw["Diastolic"] = min(float(raw["Diastolic"]), 80.0)
        controlled_raw["Smoking"] = 0.0

        prep_ctrl = prepare_serving_patient(controlled_raw)
        df_ctrl = pd.DataFrame([prep_ctrl])
        direct_ctrl_prob = float(pipe.predict_proba(df_ctrl[cols])[0, 1])

        # Verify API controlled_risk equals direct model prediction of controlled input
        assert round(api_controlled_risk, 4) == round(direct_ctrl_prob, 4)


# ==============================================================================
# TEST 5: scoreLogic.js contains no calculateRisk
# ==============================================================================
def test_scorelogic_contains_no_calculaterisk():
    scorelogic_path = ROOT / "frontend" / "src" / "data" / "scoreLogic.js"
    assert scorelogic_path.exists(), "scoreLogic.js not found!"

    content = scorelogic_path.read_text(encoding="utf-8")
    assert "calculateRisk" not in content, (
        "calculateRisk found in scoreLogic.js! The frontend must never compute risk formulas."
    )
    assert "gamma" not in content, (
        "Weibull gamma found in scoreLogic.js! All survival curves must come from the backend."
    )
