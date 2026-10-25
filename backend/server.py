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

# Load .env file for API keys
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path)
    print("[OK] .env file loaded successfully.")
except ImportError:
    print("[WARN] python-dotenv not installed. Skipping .env loading.")

# Gemini AI setup
_GEMINI_AVAILABLE = False
_GEMINI_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]

try:
    import google.generativeai as genai
    _GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if _GEMINI_API_KEY and _GEMINI_API_KEY != "your_gemini_api_key_here":
        genai.configure(api_key=_GEMINI_API_KEY)
        _GEMINI_AVAILABLE = True
        print("[OK] Gemini AI configured successfully with active models.")
    else:
        print("[WARN] GEMINI_API_KEY not set or placeholder found. AI guidance will use fallback mode.")
except ImportError:
    print("[WARN] google-generativeai not installed. AI guidance will use fallback mode.")

# Ensure programs/ and programs/macros are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROGRAMS_DIR = PROJECT_ROOT / "programs"
MACROS_DIR = PROGRAMS_DIR / "macros"

for p in [str(PROJECT_ROOT), str(PROGRAMS_DIR), str(MACROS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from macros.serving import validate_patient_input, prepare_serving_patient, normalize_sex, smoke_band
    _normalize_sex = normalize_sex
    _smoke_band = smoke_band
except ImportError:
    try:
        from serving import validate_patient_input, prepare_serving_patient, normalize_sex, smoke_band
        _normalize_sex = normalize_sex
        _smoke_band = smoke_band
    except ImportError:
        try:
            from macros.score_patient import _normalize_sex, _smoke_band
        except ImportError:
            try:
                from score_patient import _normalize_sex, _smoke_band
            except ImportError:
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

        def validate_patient_input(record):
            return True, None

        def prepare_serving_patient(raw_record):
            rec = dict(raw_record)
            rec["Sex"] = _normalize_sex(rec.get("Sex", "Female"))
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
            bmi = (weight * 703.0 / (height ** 2)) if height > 0 else 25.0
            rec["BMI"] = bmi
            rec["Smoke_Band"] = _smoke_band(smoking)
            rec["Chol_Missing"] = 0
            rec["Smoke_Missing"] = 0
            rec["Pulse_Pressure"] = systolic - diastolic
            rec["MAP"] = diastolic + (systolic - diastolic) / 3.0
            rec["Hypertension"] = int(systolic >= 140 or diastolic >= 90)
            rec["Age_x_Systolic"] = age * systolic
            return rec

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

# Cox Survival Model Loader
SURVIVAL_MODEL_PATH = MODELS_DIR / "survival_cox.pkl"
LOADED_COX = None

def get_loaded_cox():
    global LOADED_COX
    if LOADED_COX is None:
        if not SURVIVAL_MODEL_PATH.exists():
            raise FileNotFoundError(f"Survival model file {SURVIVAL_MODEL_PATH} not found.")
        cph, cols, max_fu = joblib.load(SURVIVAL_MODEL_PATH)
        LOADED_COX = (cph, cols, max_fu)
    return LOADED_COX

# Warm up models on start
for m in ["logistic", "random_forest", "gradient_boosting"]:
    try:
        get_loaded_model(m)
        print(f"[OK] Preloaded model: {m}")
    except Exception as e:
        print(f"[WARN] Could not preload {m}: {e}")

try:
    get_loaded_cox()
    print("[OK] Preloaded survival model: survival_cox")
except Exception as e:
    print(f"[WARN] Could not preload survival model: {e}")


def prepare_full_patient(raw_record):
    """Transform raw patient inputs into feature layout required by any of the 3 models."""
    return prepare_serving_patient(raw_record)


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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) or {}

        # Validate input bounds
        is_valid, err_msg = validate_patient_input(data)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400

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

        # Controlled scenario for selected model (BP <= 120/80 mmHg, Smoking = 0)
        ctrl_data = dict(data)
        ctrl_data["Systolic"] = min(float(data.get("Systolic", 120)), 120.0)
        ctrl_data["Diastolic"] = min(float(data.get("Diastolic", 80)), 80.0)
        ctrl_data["Smoking"] = 0.0
        ctrl_prep = prepare_full_patient(ctrl_data)
        X_ctrl = pd.DataFrame([ctrl_prep])
        ctrl_prob = float(pipe.predict_proba(X_ctrl[cols])[0, 1])

        band = "Low" if prob < 0.20 else "Medium" if prob < 0.50 else "High"
        flagged = bool(prob >= threshold)

        clinical = assess_clinical_metrics(data)
        insights = generate_patient_insights(data, prob, clinical)

        return jsonify({
            "success": True,
            "model_used": model_name,
            "risk": round(prob, 4),
            "risk_percentage": round(prob * 100, 1),
            "controlled_risk": round(ctrl_prob, 4),
            "controlled_risk_percentage": round(ctrl_prob * 100, 1),
            "scenario_difference_percentage": round((prob - ctrl_prob) * 100, 1),
            "scenario_label": "Illustrative scenario, not a predicted treatment effect (the model measures association, not causation)",
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


@app.route("/api/survival", methods=["POST"])
@app.route("/survival", methods=["POST"])
def survival():
    try:
        data = request.get_json(force=True) or {}

        # Validate input bounds
        is_valid, err_msg = validate_patient_input(data)
        if not is_valid:
            return jsonify({"success": False, "error": err_msg}), 400

        cph, covs, max_fu = get_loaded_cox()
        prep = prepare_serving_patient(data)
        male_val = 1 if prep["Sex"] == "Male" else 0

        # Untreated baseline
        untreated_row = pd.DataFrame([{
            "AgeAtStart": prep["AgeAtStart"],
            "Male": male_val,
            "Systolic": prep["Systolic"],
            "Cholesterol": prep["Cholesterol"],
            "BMI": prep["BMI"],
            "Smoking": prep["Smoking"]
        }])

        # Controlled scenario (Systolic <= 120, Diastolic <= 80, Smoking = 0)
        controlled_row = pd.DataFrame([{
            "AgeAtStart": prep["AgeAtStart"],
            "Male": male_val,
            "Systolic": min(prep["Systolic"], 120.0),
            "Cholesterol": prep["Cholesterol"],
            "BMI": prep["BMI"],
            "Smoking": 0.0
        }])

        time_points = [1.0, 3.0, 5.0, 10.0, 15.0, float(max_fu)]

        u_surv_df = cph.predict_survival_function(untreated_row, times=time_points)
        c_surv_df = cph.predict_survival_function(controlled_row, times=time_points)

        trajectory = []
        # Year 0 baseline
        trajectory.append({
            "year": 0,
            "untreatedSurvival": 100.0,
            "untreatedRisk": 0.0,
            "controlledSurvival": 100.0,
            "controlledRisk": 0.0,
            "difference": 0.0
        })

        for t in time_points:
            t_int = int(t) if t.is_integer() else round(t, 1)
            u_s = float(u_surv_df.loc[t].iloc[0])
            c_s = float(c_surv_df.loc[t].iloc[0])

            u_surv_pct = round(u_s * 100.0, 2)
            u_risk_pct = round((1.0 - u_s) * 100.0, 2)
            c_surv_pct = round(c_s * 100.0, 2)
            c_risk_pct = round((1.0 - c_s) * 100.0, 2)
            # Unclamped difference in scenario
            diff_pct = round(c_surv_pct - u_surv_pct, 2)

            trajectory.append({
                "year": t_int,
                "untreatedSurvival": u_surv_pct,
                "untreatedRisk": u_risk_pct,
                "controlledSurvival": c_surv_pct,
                "controlledRisk": c_risk_pct,
                "difference": diff_pct
            })

        survival_probabilities = {
            f"year_{int(t) if t.is_integer() else str(t).replace('.', '_')}": round(float(u_surv_df.loc[t].iloc[0]), 4)
            for t in time_points
        }

        return jsonify({
            "success": True,
            "model": "CoxPHFitter",
            "time_points": [int(t) if t.is_integer() else t for t in time_points],
            "survival_probabilities": survival_probabilities,
            "trajectory": trajectory,
            "note": "Survivors are assumed censored at 32 years; the dataset has no follow-up end date.",
            "disclaimer": "Illustrative scenario, not a predicted treatment effect (the model measures association, not causation)"
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


@app.route("/api/survival-trajectory", methods=["POST"])
def survival_trajectory():
    """
    Calculate dynamic Kaplan-Meier time-to-event survival trajectory comparing:
    1. Untreated baseline risk profile
    2. Controlled risk profile (Blood pressure controlled to <=120/80 mmHg and Smoking = 0)
    Evaluates at Years 1, 3, 5, 10, and 15.
    """
    try:
        data = request.get_json(force=True) or {}
        model_name = data.get("model", "logistic")

        # 1. Baseline untreated patient
        base_prepared = prepare_full_patient(data)
        X_base = pd.DataFrame([base_prepared])
        pipe, cols = get_loaded_model(model_name)
        base_prob = float(pipe.predict_proba(X_base[cols])[0, 1])

        # 2. Controlled patient (BP <= 120/80, Smoking = 0)
        controlled_data = dict(data)
        controlled_data["Systolic"] = min(float(data.get("Systolic", 120) or 120), 120.0)
        controlled_data["Diastolic"] = min(float(data.get("Diastolic", 80) or 80), 80.0)
        controlled_data["Smoking"] = 0
        ctrl_prepared = prepare_full_patient(controlled_data)
        X_ctrl = pd.DataFrame([ctrl_prepared])
        ctrl_prob = float(pipe.predict_proba(X_ctrl[cols])[0, 1])

        # Bound probabilities
        base_p10 = min(max(base_prob, 0.01), 0.98)
        ctrl_p10 = min(max(ctrl_prob, 0.01), base_p10)

        # Survival time points: 0, 1, 3, 5, 10, 15 years
        time_points = [0, 1, 3, 5, 10, 15]
        gamma = 1.32  # Weibull acceleration parameter for cardiovascular mortality
        base_H10 = -np.log(1.0 - base_p10)
        ctrl_H10 = -np.log(1.0 - ctrl_p10)

        trajectory = []
        for t in time_points:
            if t == 0:
                trajectory.append({
                    "year": 0,
                    "untreated_risk": 0.0,
                    "untreated_survival": 100.0,
                    "controlled_risk": 0.0,
                    "controlled_survival": 100.0,
                    "absolute_risk_reduction": 0.0,
                    "survival_gain": 0.0
                })
            else:
                scale = float((t / 10.0) ** gamma)
                u_risk = float(1.0 - np.exp(-base_H10 * scale))
                u_surv = float(np.exp(-base_H10 * scale))
                c_risk = float(1.0 - np.exp(-ctrl_H10 * scale))
                c_surv = float(np.exp(-ctrl_H10 * scale))

                trajectory.append({
                    "year": t,
                    "untreated_risk": round(u_risk * 100, 1),
                    "untreated_survival": round(u_surv * 100, 1),
                    "controlled_risk": round(c_risk * 100, 1),
                    "controlled_survival": round(c_surv * 100, 1),
                    "absolute_risk_reduction": round((u_risk - c_risk) * 100, 1),
                    "survival_gain": round((c_surv - u_surv) * 100, 1)
                })

        return jsonify({
            "success": True,
            "model_used": model_name,
            "baseline_10yr_risk": round(base_prob * 100, 1),
            "controlled_10yr_risk": round(ctrl_prob * 100, 1),
            "absolute_reduction_10yr": round((base_p10 - ctrl_p10) * 100, 1),
            "trajectory": trajectory
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/ai-guidance", methods=["POST"])
def ai_guidance():
    """Generate AI-powered personalized care plan using Google Gemini."""
    try:
        data = request.get_json(force=True) or {}

        age = int(float(data.get("AgeAtStart", 50) or 50))
        sex = data.get("Sex", "Female")
        systolic = int(float(data.get("Systolic", 120) or 120))
        diastolic = int(float(data.get("Diastolic", 80) or 80))
        cholesterol = int(float(data.get("Cholesterol", 200) or 200))
        weight = int(float(data.get("Weight", 160) or 160))
        height = int(float(data.get("Height", 66) or 66))
        smoking = int(float(data.get("Smoking", 0) or 0))
        risk_score = float(data.get("riskScore", 0.3))
        risk_percentage = round(risk_score * 100, 1)
        risk_band = data.get("riskBand", "Medium")

        bmi = round((weight * 703.0 / (height ** 2)) if height > 0 else 25.0, 1)

        if not _GEMINI_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Gemini AI is not configured. Please add a valid GEMINI_API_KEY (starting with AIzaSy...) to the backend/.env file.",
                "fallback": True
            }), 503

        prompt = f"""You are an expert cardiologist AI assistant generating a structured personalized care plan.

Patient Clinical Data:
- Age: {age} years
- Sex: {sex}
- Systolic Blood Pressure: {systolic} mmHg
- Diastolic Blood Pressure: {diastolic} mmHg
- Total Cholesterol: {cholesterol} mg/dL
- Weight: {weight} lbs, Height: {height} inches, BMI: {bmi}
- Smoking: {smoking} cigarettes/day
- 10-Year Cardiovascular Mortality Risk Score: {risk_percentage}% ({risk_band} Risk)

Generate a comprehensive, specific, and actionable care plan in valid JSON format with this exact structure:
{{
  "riskSummary": "2-3 sentence summary of this patient's cardiovascular profile and urgency level",
  "primaryGoal": "One clear clinical goal sentence for risk reduction",
  "dietPlan": {{
    "title": "Name of the specific diet protocol recommended",
    "sodiumTarget": "Specific sodium limit in mg/day with reason",
    "mealTiming": "Specific meal timing advice",
    "guidelines": [
      "Specific food item 1 with exact quantity and clinical reason",
      "Specific food item 2 with exact quantity and clinical reason",
      "Specific food item 3 with exact quantity and clinical reason",
      "Specific food item 4 with exact quantity and clinical reason",
      "Specific food item 5 with foods to avoid with clinical reason"
    ],
    "sampleMeal": "A specific one-day sample meal plan (breakfast, lunch, dinner, snack)"
  }},
  "exercisePlan": {{
    "title": "Name of the exercise protocol",
    "weeklyMinutes": "Total minutes per week",
    "intensity": "Target heart rate zone or RPE scale guidance",
    "schedule": [
      {{"day": "Monday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Tuesday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Wednesday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Thursday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Friday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Saturday", "activity": "Specific exercise with duration, intensity, and clinical benefit"}},
      {{"day": "Sunday", "activity": "Rest and recovery activity"}}
    ],
    "progressionTip": "How to gradually increase intensity over 8-12 weeks"
  }},
  "lifestyleChanges": [
    "Specific sleep recommendation with reason",
    "Specific stress management technique with duration/frequency",
    "Specific smoking or other habit change if relevant"
  ],
  "medicalGuidance": [
    "Doctor-facing: specific medication class recommendation with clinical rationale",
    "Specific lab tests or monitoring to order with frequency",
    "Follow-up visit schedule recommendation"
  ],
  "targetBiomarkers": {{
    "bpTarget": "Specific BP target in mmHg",
    "cholesterolTarget": "Specific cholesterol target in mg/dL",
    "bmiTarget": "Specific BMI target range",
    "timelineWeeks": "Realistic weeks to achieve targets with adherence"
  }}
}}

IMPORTANT: Return ONLY valid JSON. No markdown code blocks, no explanations, just the JSON object."""

        response = None
        used_model = None
        last_error = None

        for model_name in _GEMINI_MODELS:
            try:
                gen_model = genai.GenerativeModel(model_name)
                response = gen_model.generate_content(prompt)
                if response and response.text:
                    used_model = model_name
                    break
            except Exception as model_err:
                last_error = model_err
                print(f"[WARN] Gemini model {model_name} failed: {model_err}. Trying fallback...")
                continue

        if not response or not used_model:
            raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")

        raw_text = response.text.strip()

        # Clean markdown code fences if model wrapped response in ```json ... ```
        if "```" in raw_text:
            # Extract content between first ``` and last ```
            start_idx = raw_text.find("```")
            end_idx = raw_text.rfind("```")
            inner = raw_text[start_idx:end_idx].strip()
            # remove language identifier if present like ```json
            if inner.startswith("```json"):
                inner = inner[7:].strip()
            elif inner.startswith("```"):
                inner = inner[3:].strip()
            raw_text = inner

        care_plan = json.loads(raw_text)

        return jsonify({
            "success": True,
            "source": used_model,
            "patientSummary": {
                "age": age, "sex": sex, "systolic": systolic, "diastolic": diastolic,
                "cholesterol": cholesterol, "bmi": bmi, "smoking": smoking,
                "riskPercentage": risk_percentage, "riskBand": risk_band
            },
            "carePlan": care_plan
        })

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"Gemini returned invalid JSON: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/copilot", methods=["POST"])
def clinical_copilot_endpoint():
    """Embedded Clinical AI Copilot for physician queries."""
    try:
        from clinical_copilot import answer_clinical_query
        payload = request.get_json(force=True) or {}
        message = payload.get("message", "").strip()
        if not message:
            return jsonify({"success": False, "error": "Query message cannot be empty."}), 400

        patient = payload.get("patient") or {}
        history = payload.get("history") or []

        reply = answer_clinical_query(message, patient, history)
        return jsonify({
            "success": True,
            "reply": reply,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Heart Risk Flask API server on http://127.0.0.1:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
