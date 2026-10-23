import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

# Ensure programs/ and programs/macros are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRAMS_DIR = PROJECT_ROOT / "programs"
sys.path.insert(0, str(PROGRAMS_DIR))

from macros.score_patient import _normalize_sex, _smoke_band

app = Flask(__name__)
CORS(app)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
SAVED_PATIENTS_JSON = PROJECT_ROOT / "data" / "processed" / "saved_patients.json"
SAVED_PATIENTS_CSV = PROJECT_ROOT / "data" / "processed" / "saved_patients.csv"


def load_saved_patients():
    if not SAVED_PATIENTS_JSON.exists():
        return []
    try:
        with open(SAVED_PATIENTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load saved patients: {e}")
        return []


def persist_saved_patients(patients):
    SAVED_PATIENTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SAVED_PATIENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

    # Also sync to CSV for direct dataset/tabular inspection
    if patients:
        df = pd.DataFrame(patients)
        df.to_csv(SAVED_PATIENTS_CSV, index=False)
    elif SAVED_PATIENTS_CSV.exists():
        SAVED_PATIENTS_CSV.unlink(missing_ok=True)


# Cache loaded pipelines in memory for fast scoring
LOADED_MODELS = {}

def get_loaded_model(model_name="logistic"):
    if model_name not in LOADED_MODELS:
        path = MODELS_DIR / f"{model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found.")
        pipe, cols = joblib.load(path)
        LOADED_MODELS[model_name] = (pipe, cols)
    return LOADED_MODELS[model_name]

# Warm up models on start
for m in ["logistic", "random_forest", "gradient_boosting"]:
    try:
        get_loaded_model(m)
        print(f"[OK] Preloaded model: {m}")
    except Exception as e:
        print(f"[WARN] Could not preload {m}: {e}")


def prepare_full_patient(raw_record):
    """Transform raw patient inputs into feature layout required by any of the 3 models."""
    rec = dict(raw_record)

    # Normalize sex
    rec["Sex"] = _normalize_sex(rec.get("Sex", "Female"))

    # Numeric conversions with safe fallbacks
    age = float(rec.get("AgeAtStart", 50) or 50)
    systolic = float(rec.get("Systolic", 120) or 120)
    diastolic = float(rec.get("Diastolic", 80) or 80)
    smoking = float(rec.get("Smoking", 0) or 0)
    weight = float(rec.get("Weight", 160) or 160)
    height = float(rec.get("Height", 66) or 66)
    cholesterol = float(rec.get("Cholesterol", 200) or 200)

    rec["AgeAtStart"] = age
    rec["Systolic"] = systolic
    rec["Diastolic"] = diastolic
    rec["Smoking"] = smoking
    rec["Cholesterol"] = cholesterol

    # BMI
    bmi = (weight * 703.0 / (height ** 2)) if height > 0 else 25.0
    rec["BMI"] = bmi

    # Smoke band
    rec["Smoke_Band"] = _smoke_band(smoking)

    # Missing flags
    rec["Chol_Missing"] = 0
    rec["Smoke_Missing"] = 0

    # Engineered clinical features for RF and Gradient Boosting
    rec["Pulse_Pressure"] = systolic - diastolic
    rec["MAP"] = diastolic + (systolic - diastolic) / 3.0
    rec["Hypertension"] = int(systolic >= 140 or diastolic >= 90)
    rec["Age_x_Systolic"] = age * systolic

    return rec


def assess_clinical_metrics(record):
    """Compute standard clinical categorizations for patient display."""
    weight = float(record.get("Weight", 160) or 160)
    height = float(record.get("Height", 66) or 66)
    bmi = (weight * 703.0 / (height ** 2)) if height > 0 else 25.0
    
    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25.0:
        bmi_status = "Healthy weight"
    elif bmi < 30.0:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    systolic = float(record.get("Systolic", 120) or 120)
    diastolic = float(record.get("Diastolic", 80) or 80)
    pulse_pressure = systolic - diastolic

    if systolic < 120 and diastolic < 80:
        bp_status = "Optimal / Normal"
    elif systolic <= 129 and diastolic < 80:
        bp_status = "Elevated Blood Pressure"
    elif systolic <= 139 or diastolic <= 89:
        bp_status = "Stage 1 Hypertension"
    elif systolic <= 180 or diastolic <= 120:
        bp_status = "Stage 2 Hypertension"
    else:
        bp_status = "Hypertensive Crisis"

    cholesterol = float(record.get("Cholesterol", 200) or 200)
    if cholesterol < 200:
        chol_status = "Desirable (<200 mg/dL)"
    elif cholesterol < 240:
        chol_status = "Borderline High (200-239 mg/dL)"
    else:
        chol_status = "High (>=240 mg/dL)"

    smoking = float(record.get("Smoking", 0) or 0)
    if smoking == 0:
        smoke_status = "Non-smoker"
    elif smoking < 15:
        smoke_status = f"Light Smoker ({int(smoking)} cigs/day)"
    elif smoking < 25:
        smoke_status = f"Moderate Smoker ({int(smoking)} cigs/day)"
    else:
        smoke_status = f"Heavy Smoker ({int(smoking)}+ cigs/day)"

    return {
        "bmi": round(bmi, 1),
        "bmi_status": bmi_status,
        "pulse_pressure": int(pulse_pressure),
        "bp_status": bp_status,
        "chol_status": chol_status,
        "smoke_status": smoke_status,
    }


def generate_patient_insights(record, risk_prob, clinical):
    """Generate human-readable factors and recommendations for patient communication."""
    elevating_factors = []
    protective_factors = []
    actions = []

    age = float(record.get("AgeAtStart", 50) or 50)
    systolic = float(record.get("Systolic", 120) or 120)
    diastolic = float(record.get("Diastolic", 80) or 80)
    cholesterol = float(record.get("Cholesterol", 200) or 200)
    smoking = float(record.get("Smoking", 0) or 0)
    bmi = clinical["bmi"]

    # Age impact
    if age >= 55:
        elevating_factors.append(f"Age ({int(age)} yrs): Risk increases past age 50 due to vascular stiffening.")
    elif age < 45:
        protective_factors.append(f"Age ({int(age)} yrs): Younger demographic provides cardiovascular resilience.")

    # Blood Pressure impact
    if systolic >= 140 or diastolic >= 90:
        elevating_factors.append(
            f"Blood Pressure ({int(systolic)}/{int(diastolic)} mmHg - {clinical['bp_status']}): Elevates heart workload and mortality hazard."
        )
        actions.append("Initiate or review antihypertensive therapy; aim for target BP under 130/80 mmHg.")
    elif systolic < 120 and diastolic < 80:
        protective_factors.append(f"Blood Pressure ({int(systolic)}/{int(diastolic)} mmHg): Optimal reading protects arteries.")

    # Smoking impact
    if smoking > 0:
        elevating_factors.append(
            f"Smoking ({clinical['smoke_status']}): Tobacco exposure damages vascular endothelium and more than doubles risk."
        )
        actions.append("Enroll in a structured smoking cessation program (nicotine replacement, counseling).")
    else:
        protective_factors.append("Smoking (Non-smoker): Zero tobacco exposure significantly preserves arterial elasticity.")

    # Cholesterol impact
    if cholesterol >= 240:
        elevating_factors.append(f"Total Cholesterol ({int(cholesterol)} mg/dL): High serum lipids accelerate coronary plaque buildup.")
        actions.append("Perform full fasting lipid panel (LDL, HDL, Triglycerides); discuss statin therapy with your physician.")
    elif cholesterol < 200:
        protective_factors.append(f"Cholesterol ({int(cholesterol)} mg/dL): Within desirable limits (<200 mg/dL).")

    # BMI impact
    if bmi >= 30.0:
        elevating_factors.append(f"BMI ({bmi} - {clinical['bmi_status']}): Contributes to metabolic syndrome and cardiac strain.")
        actions.append("Target a 5-10% gradual weight reduction through low-glycemic, Mediterranean-style nutrition.")
    elif 18.5 <= bmi <= 24.9:
        protective_factors.append(f"BMI ({bmi}): In the healthy target range.")

    # Urgency & summary
    if risk_prob >= 0.50:
        urgency = "High Priority Clinical Alert"
        summary_message = "Your calculated cardiovascular mortality risk profile is in the High tier (>50%). Prompt consultation with a healthcare provider is strongly recommended to review preventive cardiology interventions."
    elif risk_prob >= 0.20:
        urgency = "Moderate Preventive Review"
        summary_message = "Your cardiovascular risk is in the Medium tier (20%-50%). Targeted lifestyle adjustments and proactive blood pressure/lipid control can effectively prevent high-risk escalation."
    else:
        urgency = "Low Risk / Maintenance"
        summary_message = "Your clinical profile falls in the Low risk tier (<20%). Maintain routine wellness checkups, balanced nutrition, and regular aerobic activity."

    if not actions:
        actions.append("Maintain 150 minutes of moderate aerobic physical activity per week.")
        actions.append("Continue routine preventive annual wellness screenings.")

    return {
        "urgency": urgency,
        "summary": summary_message,
        "elevating_factors": elevating_factors,
        "protective_factors": protective_factors,
        "recommended_actions": actions,
    }


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Heart Risk Prediction API",
        "champion_model": "logistic",
        "available_models": list(LOADED_MODELS.keys()),
        "threshold": 0.20
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}
        model_name = data.get("model", "logistic")
        threshold = float(data.get("threshold", 0.20))

        prepared = prepare_full_patient(data)
        X = pd.DataFrame([prepared])

        # Selected model
        pipe, cols = get_loaded_model(model_name)
        prob = float(pipe.predict_proba(X[cols])[0, 1])

        # Cross-model consensus
        consensus = {}
        for m_name in ["logistic", "random_forest", "gradient_boosting"]:
            try:
                m_pipe, m_cols = get_loaded_model(m_name)
                m_prob = float(m_pipe.predict_proba(X[m_cols])[0, 1])
                consensus[m_name] = round(m_prob, 4)
            except Exception as e:
                print(f"Error evaluating {m_name}: {e}")

        band = "Low" if prob < 0.20 else "Medium" if prob < 0.50 else "High"
        flagged = bool(prob >= threshold)

        clinical = assess_clinical_metrics(data)
        insights = generate_patient_insights(data, prob, clinical)

        return jsonify({
            "success": True,
            "model_used": model_name,
            "risk": round(prob, 4),
            "risk_percentage": round(prob * 100, 1),
            "risk_band": band,
            "flagged": flagged,
            "threshold": threshold,
            "consensus": consensus,
            "clinical_metrics": clinical,
            "insights": insights,
            "features_processed": {
                "AgeAtStart": prepared.get("AgeAtStart"),
                "Sex": prepared.get("Sex"),
                "Systolic": prepared.get("Systolic"),
                "Diastolic": prepared.get("Diastolic"),
                "Cholesterol": prepared.get("Cholesterol"),
                "BMI": round(prepared.get("BMI", 0), 1),
                "Smoke_Band": prepared.get("Smoke_Band"),
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/presets", methods=["GET"])
def presets():
    sample_patients = [
        {
            "id": "high_risk",
            "name": "High Risk (Hypertensive Smoker)",
            "description": "58-year-old male with Stage 2 Hypertension, elevated cholesterol, and moderate smoking.",
            "data": {
                "AgeAtStart": 58,
                "Sex": "Male",
                "Systolic": 165,
                "Diastolic": 96,
                "Cholesterol": 248,
                "Weight": 195,
                "Height": 68,
                "Smoking": 20
            }
        },
        {
            "id": "moderate_risk",
            "name": "Moderate Risk (Stage 1 Borderline)",
            "description": "52-year-old female with mildly elevated blood pressure and borderline cholesterol.",
            "data": {
                "AgeAtStart": 52,
                "Sex": "Female",
                "Systolic": 138,
                "Diastolic": 86,
                "Cholesterol": 215,
                "Weight": 155,
                "Height": 65,
                "Smoking": 5
            }
        },
        {
            "id": "low_risk",
            "name": "Low Risk (Healthy Profile)",
            "description": "35-year-old female, non-smoker, normal blood pressure and lipid profile.",
            "data": {
                "AgeAtStart": 35,
                "Sex": "Female",
                "Systolic": 115,
                "Diastolic": 76,
                "Cholesterol": 175,
                "Weight": 130,
                "Height": 65,
                "Smoking": 0
            }
        },
        {
            "id": "elderly_normotensive",
            "name": "Senior Profile (Controlled BP)",
            "description": "68-year-old male, non-smoker with well-managed blood pressure.",
            "data": {
                "AgeAtStart": 68,
                "Sex": "Male",
                "Systolic": 128,
                "Diastolic": 82,
                "Cholesterol": 190,
                "Weight": 170,
                "Height": 70,
                "Smoking": 0
            }
        }
    ]
    return jsonify(sample_patients)


@app.route("/api/reports/summary", methods=["GET"])
def reports_summary():
    odds_path = REPORTS_DIR / "odds_ratios.csv"
    odds_data = []
    if odds_path.exists():
        df = pd.read_csv(odds_path)
        odds_data = df.to_dict(orient="records")

    scorecard_path = REPORTS_DIR / "scorecard_points.csv"
    scorecard_data = []
    if scorecard_path.exists():
        df_sc = pd.read_csv(scorecard_path)
        scorecard_data = df_sc.to_dict(orient="records")

    return jsonify({
        "odds_ratios": odds_data,
        "scorecard": scorecard_data,
        "model_performance": {
            "champion": "Logistic Regression",
            "validation_auc": 0.779,
            "brier_score": 0.165,
            "threshold": 0.20
        }
    })


@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Retrieve all saved patient records."""
    patients = load_saved_patients()
    return jsonify({
        "success": True,
        "patients": patients,
        "total": len(patients)
    })


@app.route("/api/patients/save", methods=["POST"])
def save_patient():
    """Save a patient record with vitals, prediction output, and timestamp."""
    try:
        data = request.get_json(force=True) or {}
        patient_name = data.get("name") or data.get("patient_name") or f"Patient-{uuid.uuid4().hex[:6].upper()}"

        # Numeric conversions with safe fallbacks
        age = float(data.get("AgeAtStart", 50) or 50)
        sex = _normalize_sex(data.get("Sex", "Female"))
        systolic = float(data.get("Systolic", 120) or 120)
        diastolic = float(data.get("Diastolic", 80) or 80)
        cholesterol = float(data.get("Cholesterol", 200) or 200)
        weight = float(data.get("Weight", 160) or 160)
        height = float(data.get("Height", 66) or 66)
        smoking = float(data.get("Smoking", 0) or 0)
        notes = str(data.get("notes", "")).strip()
        model_used = data.get("model", "logistic")

        bmi = round((weight * 703.0 / (height ** 2)) if height > 0 else 25.0, 1)

        # Get or compute prediction
        risk = data.get("risk")
        if risk is None:
            prepared = prepare_full_patient(data)
            X = pd.DataFrame([prepared])
            pipe, cols = get_loaded_model(model_used)
            risk = float(pipe.predict_proba(X[cols])[0, 1])
        else:
            risk = float(risk)

        risk_percentage = round(risk * 100, 1)
        risk_band = "Low" if risk < 0.20 else "Medium" if risk < 0.50 else "High"

        clinical = assess_clinical_metrics(data)

        record = {
            "id": f"pat_{uuid.uuid4().hex[:8]}",
            "name": patient_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "AgeAtStart": int(age),
            "Sex": sex,
            "Systolic": int(systolic),
            "Diastolic": int(diastolic),
            "Cholesterol": int(cholesterol),
            "Weight": int(weight),
            "Height": int(height),
            "Smoking": int(smoking),
            "BMI": bmi,
            "bp_status": clinical["bp_status"],
            "chol_status": clinical["chol_status"],
            "smoke_status": clinical["smoke_status"],
            "model_used": model_used,
            "risk": round(risk, 4),
            "risk_percentage": risk_percentage,
            "risk_band": risk_band,
            "notes": notes
        }

        patients = load_saved_patients()
        patients.insert(0, record)  # Newest first
        persist_saved_patients(patients)

        return jsonify({
            "success": True,
            "message": f"Patient record for '{patient_name}' saved successfully.",
            "patient": record,
            "total": len(patients)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/patients/<patient_id>", methods=["DELETE"])
def delete_patient(patient_id):
    """Delete a saved patient record."""
    try:
        patients = load_saved_patients()
        filtered = [p for p in patients if p.get("id") != patient_id]
        if len(filtered) == len(patients):
            return jsonify({"success": False, "error": "Record not found"}), 404

        persist_saved_patients(filtered)
        return jsonify({
            "success": True,
            "message": "Record deleted successfully.",
            "deleted_id": patient_id,
            "total": len(filtered)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Heart Risk Flask API server on http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
