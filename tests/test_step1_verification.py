"""
Step 1 Verification:
Compare direct output from models/logistic.pkl with Flask /predict endpoint
for 5 representative patients.
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "programs"))

from backend.server import app
from programs.macros.serving import prepare_serving_patient

PATIENTS = [
    {
        "name": "Patient 1 (Sample: 58F, 145/88, Chol 210, 160lb, 66in, 12 cigs)",
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
        "name": "Patient 2 (Young non-smoker: 22F, 110/70, Chol 170, 125lb, 64in, 0 cigs)",
        "data": {
            "AgeAtStart": 22,
            "Sex": "Female",
            "Systolic": 110,
            "Diastolic": 70,
            "Cholesterol": 170,
            "Weight": 125,
            "Height": 64,
            "Smoking": 0,
        }
    },
    {
        "name": "Patient 3 (Old heavy male smoker: 68M, 160/96, Chol 255, 200lb, 70in, 30 cigs)",
        "data": {
            "AgeAtStart": 68,
            "Sex": "Male",
            "Systolic": 160,
            "Diastolic": 96,
            "Cholesterol": 255,
            "Weight": 200,
            "Height": 70,
            "Smoking": 30,
        }
    },
    {
        "name": "Patient 4 (Middle-aged male moderate smoker: 52M, 138/86, Chol 215, 175lb, 69in, 15 cigs)",
        "data": {
            "AgeAtStart": 52,
            "Sex": "Male",
            "Systolic": 138,
            "Diastolic": 86,
            "Cholesterol": 215,
            "Weight": 175,
            "Height": 69,
            "Smoking": 15,
        }
    },
    {
        "name": "Patient 5 (Senior female non-smoker: 72F, 142/82, Chol 230, 145lb, 63in, 0 cigs)",
        "data": {
            "AgeAtStart": 72,
            "Sex": "Female",
            "Systolic": 142,
            "Diastolic": 82,
            "Cholesterol": 230,
            "Weight": 145,
            "Height": 63,
            "Smoking": 0,
        }
    },
]


def run_verification():
    # 1. Load raw model pipeline directly
    model_path = ROOT / "models" / "logistic.pkl"
    pipe, cols = joblib.load(model_path)

    # 2. Setup Flask test client
    client = app.test_client()

    print("=" * 85)
    print(f"{'Patient':<45} | {'Direct Model':<12} | {'/predict API':<12} | {'Match':<6}")
    print("=" * 85)

    all_matched = True
    for p in PATIENTS:
        name = p["name"]
        raw = p["data"]

        # A. Direct pipeline score
        prep = prepare_serving_patient(raw)
        df = pd.DataFrame([prep])
        direct_prob = float(pipe.predict_proba(df[cols])[0, 1])

        # B. Endpoint score via Flask client
        res = client.post("/predict", json=raw)
        assert res.status_code == 200, f"Endpoint failed: {res.get_data(as_text=True)}"
        api_data = json.loads(res.get_data(as_text=True))
        api_risk = float(api_data["risk"])

        # Compare to 4 decimal places
        direct_4d = round(direct_prob, 4)
        api_4d = round(api_risk, 4)
        match = (direct_4d == api_4d)
        if not match:
            all_matched = False

        print(f"{name[:45]:<45} | {direct_4d:<12.4f} | {api_4d:<12.4f} | {str(match):<6}")

    print("=" * 85)
    if all_matched:
        print("[SUCCESS] All 5 patients matched to 4 decimal places between direct model & API!")
    else:
        print("[FAILURE] Some patients differed!")
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
