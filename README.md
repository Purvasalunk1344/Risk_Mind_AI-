# ❤️ Heart Disease Mortality Risk Prediction System & Clinical Studio

An end-to-end clinical machine learning and decision-support system trained on the **Framingham Heart Study** cohort (`SASHELP.HEART`). The system predicts **10-year cardiovascular mortality risk**, calculates dynamic time-to-event Kaplan-Meier survival curves, explains patient-level risk drivers, evaluates multi-model algorithm consensus, monitors data drift, and provides an interactive clinical web studio with generative AI clinical copilot assistance.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Features](#-key-features)
- [Full-Stack System Architecture](#-full-stack-system-architecture)
- [What the Frontend Tells the User](#-what-the-frontend-tells-the-user)
  - [1. Risk Verdict & Multi-Model Consensus](#1-how-safe-or-at-risk-is-my-heart-the-risk-verdict)
  - [2. Vitals & Hemodynamics Diagnosis](#2-how-healthy-are-my-individual-body-measurements-vitals-panel)
  - [3. Explainability & Factor Attribution](#3-why-is-my-score-so-high-or-low-explainability)
  - [4. Dynamic Kaplan-Meier Survival Trajectory Curves](#4-what-is-my-time-to-event-trajectory-survival-curves)
  - [5. Patient Action Plan & Counterfactual Simulation](#5-what-should-i-do-right-now-action-plan--counterfactuals)
  - [6. Embedded Clinical AI Copilot & Gemini Care Plan](#6-generative-clinical-ai-copilot--dietexercise-protocol)
  - [7. Patient Record Vault & History](#7-patient-record-vault--longitudinal-tracking)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Project Directory Layout](#-project-directory-layout)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Run the Project](#-how-to-run-the-project)
- [API Reference](#-api-reference)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🩺 Executive Summary

Cardiovascular disease remains the leading cause of global mortality. Early, accurate prognostic assessment enables proactive interventions that prevent fatal cardiac events before irreversible damage occurs.

This repository delivers:
1. **A 10-Stage Machine Learning Pipeline**: Covering exploratory data analysis, data hygiene, hemodynamic feature engineering, train/val/test stratified splitting, model training, survival analysis (Kaplan-Meier & Cox Proportional Hazards), explainable AI (SHAP & Odds Ratios), clinical scorecard derivation, fairness auditing, and MLOps drift monitoring.
2. **A Python REST API Backend (`Flask`)**: Serves real-time inference, pre-caches trained scikit-learn models, performs Weibull time-to-event survival trajectory simulations across 15-year horizons, manages patient records, and interfaces with generative AI (Google Gemini & Groq).
3. **An Interactive Clinical Web Studio (`React + Vite`)**: A dark-mode, glassmorphic decision-support application where clinicians and patients explore risk profiles, view dynamic survival projections, compare counterfactual interventions, chat with an intelligent clinical copilot, and download comprehensive clinical PDF reports.

---

## ✨ Key Features

- **Multi-Model Algorithmic Intelligence**: Simultaneously evaluates patients against three distinct machine learning algorithms:
  - 🏆 **Champion**: Logistic Regression with standardized scaling (Validation ROC-AUC: `0.779`, Brier: `0.165`)
  - **Random Forest Classifier**: Ensemble decision tree model (`0.768` AUC)
  - **Gradient Boosting Classifier**: Sequentially optimized gradient trees (`0.774` AUC)
- **Dynamic Time-to-Event Survival Trajectory Curves (NEW)**:
  - Interactive Kaplan-Meier & Weibull-scaled survival trajectory modeled at **Years 1, 3, 5, 10, and 15**.
  - Direct side-by-side comparison of **Untreated Baseline Trajectory** vs. **Controlled Trajectory** (Blood pressure controlled to $\le 120/80\text{ mmHg}$ and smoking stopped).
  - Live metric readouts for **Absolute Risk Reduction (ARR)** and **Survival Probability Gain (%)**.
  - Toggle between **Cumulative Mortality Hazard (%)** and **Survival Probability (%)**.
- **Generative Clinical AI Copilot & Gemini Care Plan (NEW)**:
  - Embedded conversational AI copilot (powered by Groq / Google Gemini) answering patient-specific inquiries with context-aware physiological citations.
  - Comprehensive 7-day personalized dietary, aerobic/resistance exercise, and lifestyle prescription generation.
- **Explainable AI (XAI)**:
  - Statistical **Odds Ratios** with 95% Confidence Intervals via `statsmodels`
  - **SHAP (SHapley Additive exPlanations)** values and risk factor attribution
  - **Permutation Feature Importance**
- **Fairness & MLOps Governance**:
  - Subgroup fairness checks across biological sex and age demographics
  - Population Stability Index (PSI) drift monitoring for production deployments
- **Longitudinal Patient Vault**:
  - Save, reload, and manage patient profiles with automatic JSON and CSV two-way synchronization.
  - Instant 1-click loading of diverse clinical archetypes (High-Risk Hypertensive Smoker, Moderate Risk, Low Risk / Healthy, Senior Profile).

---

## 🏛️ Full-Stack System Architecture

```
+-----------------------------------------------------------------------------------------+
|                               FRAMINGHAM COHORT DATA                                    |
|                                  (SASHELP.HEART)                                        |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                            10-STAGE PYTHON ML PIPELINE                                  |
|   01_eda -> 02_cleaning -> 03_feature_engineering -> 04_split -> 05_modeling            |
|   -> 06_evaluation -> 07_survival -> 08_explainability -> 09_fairness -> 10_mlops        |
+--------------------------------------------+--------------------------------------------+
                                             | Generates
                                             v
+-----------------------------------------------------------------------------------------+
|                            MODELS & ANALYTICS ARTIFACTS                                 |
|   models/logistic.pkl | models/random_forest.pkl | models/gradient_boosting.pkl          |
|   output/plots/*.png  | output/reports/*.csv     | monitoring/drift_summary.json         |
+--------------------------------------------+--------------------------------------------+
                                             | Preloaded by
                                             v
+-----------------------------------------------------------------------------------------+
|                            FLASK REST API BACKEND (Port 5000)                           |
|   - Pre-cached Scikit-Learn Pipelines          - Derived Hemodynamic Math               |
|   - Multi-Model Consensus Scoring              - Time-to-Event Survival Trajectory Math |
|   - Patient Profile Vault (JSON & CSV Sync)    - Embedded Gemini / Groq Clinical Copilot|
+--------------------------------------------+--------------------------------------------+
                                             ^ Reverse Proxy (/api/*)
                                             |
+--------------------------------------------+--------------------------------------------+
|                         REACT + VITE CLINICAL STUDIO (Port 5173)                        |
|   - Live Mortality Risk Gauge                  - Vitals & Hemodynamics Diagnosis        |
|   - Multi-Model Consensus Bar                  - Explainability Factor Attribution      |
|   - Dynamic Kaplan-Meier Trajectory Curve      - Counterfactual Risk Simulation         |
|   - Generative Clinical AI Copilot & Care Plan - Longitudinal Patient Record Vault      |
+-----------------------------------------------------------------------------------------+
```

---

## 🖥️ What the Frontend Tells the User

The frontend acts as an empathetic **digital cardiology consultant**, translating complex statistical models into actionable, clinically grounded answers:

```
+-----------------------------------------------------------------------------------------------------+
|  HeartRisk Studio                     [ 🟢 Python ML Backend (Port 5000) ]       [ Model Selector ]  |
|  [ Presets: High-Risk Smoker | Moderate Risk | Healthy / Low Risk | Senior Profile | Blank Form ]   |
+-----------------------------------------------------------------------------------------------------+
|  HERO RESULTS & CONSENSUS PANEL:                                                                    |
|  - 10-Year Mortality Hazard Gauge: e.g., 91.3% (HIGH RISK TIER)                                     |
|  - Multi-Model Consensus: Logistic (91.3%) | Random Forest (92.1%) | Gradient Boosting (92.9%)      |
+-----------------------------------------------------------------------------------------------------+
|  DYNAMIC KAPLAN-MEIER SURVIVAL TRAJECTORY CURVE:                                                    |
|  - Baseline Untreated vs. Controlled Intervention Curve across Year 1, 3, 5, 10, and 15            |
|  - Absolute Risk Reduction (ARR) | Survival Gain (%) | Interactive Milestone Timeline Selector      |
+-----------------------------------------------------------------------------------------------------+
|  [ 📋 Patient Vitals ]  [ 🔍 Why This Risk? ]  [ 💡 Action Plan ]  [ 🤖 Copilot ]  [ 📁 Records ]   |
+-----------------------------------------------------------------------------------------------------+
```

### 1. "How safe or at-risk is my heart?" (The Risk Verdict)
- **10-Year Mortality Hazard**: High-visibility percentage gauge (e.g., **`91.3%`**).
- **Risk Tier Badge**:
  - 🟢 **Low Risk (< 20%)**: Heart profile is within safe parameters; routine maintenance.
  - 🟡 **Medium Risk (20% – 50%)**: Borderline warning; proactive lifestyle and BP management required.
  - 🔴 **High Risk (> 50%)**: Clinical alert; requires prompt cardiology evaluation.
- **Algorithm Consensus**: Displays predictions from Logistic Regression, Random Forest, and Gradient Boosting side-by-side to highlight model agreement.

### 2. "How healthy are my individual body measurements?" (Vitals Panel)
- **Blood Pressure Categorization**: Classifies readings into *Normal*, *Elevated*, *Stage 1 Hypertension*, or *Stage 2 Hypertension* (e.g. `165/96 mmHg`).
- **Pulse Pressure**: Calculates arterial stiffness ($\text{Systolic} - \text{Diastolic}$, normal 40–60 mmHg).
- **Mean Arterial Pressure (MAP)**: Approximates perfusion pressure ($\text{Diastolic} + \frac{1}{3}\Delta\text{BP}$).
- **Body Mass Index (BMI)**: Computes BMI from height and weight ($\text{BMI} = \frac{\text{Weight} \times 703}{\text{Height}^2}$) with clinical categorization (*Healthy*, *Overweight*, or *Obese*).
- **Cholesterol Status**: Classifies serum lipids (*Desirable <200*, *Borderline*, *High >=240 mg/dL*).
- **Tobacco Exposure**: Categorizes smoking severity (*Non-smoker*, *Light*, *Moderate*, *Heavy*).

### 3. "Why is my score so high (or low)?" (Explainability)
- **⚠️ Primary Risk Elevators**: Pinpoints specific variables driving up risk:
  - *"Blood Pressure (165/96 mmHg - Stage 2 Hypertension): Strains arterial walls and multiplies mortality hazard."*
  - *"Smoking (20 cigs/day): Tobacco toxins damage vascular endothelium and double mortality odds."*
  - *"Age (58 yrs): Risk increases past age 50 due to natural vascular stiffening."*
- **🛡️ Protective Factors**: Highlights positive metrics keeping the patient safe:
  - *"Non-smoker: Zero tobacco exposure preserves arterial elasticity."*
  - *"BMI (22.5): Within healthy target range."*

### 4. "What is my time-to-event trajectory?" (Survival Curves)
- **Longitudinal Projection**: Displays continuous survival probability and cumulative mortality hazard from Year 0 up to Year 15.
- **Two Comparative Curves**:
  1. 🔴 **Untreated Baseline Trajectory**: The projected trajectory based on the patient's current clinical measurements.
  2. 🟢 **Controlled Intervention Trajectory**: The simulated trajectory assuming blood pressure is managed to $\le 120/80\text{ mmHg}$ and tobacco exposure is eliminated.
- **Interactive Milestone Selector**: Click on Year 1, 3, 5, 10, or 15 to inspect:
  - Baseline Risk vs. Controlled Risk
  - **Absolute Risk Reduction (ARR)** (e.g., `-38.4%` at Year 10)
  - **Survival Gain** (e.g., `+38.4%` increased survival probability)
- **Shaded Therapeutic Benefit Area**: Highlights the preventable mortality window directly on the SVG canvas.

### 5. "What should I do right now?" (Action Plan & Counterfactuals)
- **Counterfactual Risk Simulator**: Demonstrates immediate quantifiable impact:
  - *"If you normalize blood pressure and stop smoking, your 10-year risk drops from 91.3% to 48.7% (a 42.6% absolute risk reduction)."*
- **Concrete Patient Target Milestones**:
  - Target Blood Pressure: **< 120/80 mmHg**
  - Target Cholesterol: **< 200 mg/dL**
  - Target Tobacco Use: **0 cigarettes/day**
  - Target Physical Activity: **150 min/week aerobic exercise**

### 6. Generative Clinical AI Copilot & Diet/Exercise Protocol
- **Physician & Patient AI Copilot**: Ask specific clinical questions (*"What are the pharmacological indications for this patient's BP?"*, *"What dietary changes reduce arterial stiffness?"*).
- **Structured 7-Day Care Plan**: Generates a day-by-day workout regimen, sodium targets, tailored dietary guidance, and clinical follow-up timelines via Gemini AI.

### 7. Patient Record Vault & Longitudinal Tracking
- **Saved Profiles**: Save evaluated patient records with custom clinical notes.
- **Reload & Retest**: 1-click loading from the saved patients drawer to compare progression over time.
- **Download Medical Report**: Printable PDF/report summary for electronic health records (EHR).

---

## 🔬 Machine Learning Pipeline

The pipeline is organized into sequential, modular Python scripts located in [`programs/`](file:///c:/Data%20analytics/Project/heart_Risk_Prediction_Project/programs):

| Step | Script | Description |
| :--- | :--- | :--- |
| **00** | `00_export_raw.sas` | SAS script exporting `SASHELP.HEART` into `data/raw/heart_raw.csv`. |
| **01** | `01_eda.py` | Exploratory data analysis; generates correlation heatmaps and feature distributions. |
| **02** | `02_cleaning.py` | Data hygiene; handles missing values, categorical normalization, and outlier management. |
| **03** | `03_feature_engineering.py` | Creates clinical features: `BMI`, `Pulse_Pressure`, `MAP`, `Hypertension`, `Age_x_Systolic`, `Smoke_Band`. |
| **04** | `04_split.py` | Stratified 70/15/15 train/validation/test split preserving mortality event ratios. |
| **05** | `05_modeling.py` | Trains Logistic Regression, Random Forest, and Gradient Boosting pipelines. |
| **06** | `06_evaluation.py` | Generates ROC-AUC curves, calibration plots, Brier scores, and cost-threshold optimization. |
| **07** | `07_survival.py` | Computes Kaplan-Meier survival curves and Cox Proportional Hazards regression using `lifelines`. |
| **08** | `08_explainability.py` | Calculates statsmodels Odds Ratios, SHAP values, and permutation feature importances. |
| **09** | `09_scorecard_fairness.py` | Builds a point-based clinical scorecard and evaluates demographic parity across sex/age groups. |
| **10** | `10_mlops_monitoring.py` | Computes Population Stability Index (PSI) drift benchmarks between train and test distributions. |
| **All**| `run_all.py` | Master orchestrator script that executes steps 01 through 10 sequentially in a single command. |
| **CLI**| `11_scoring_deploy.py` | Command-line scoring tool for single patient JSON or batch CSV scoring. |

---

## 📁 Project Directory Layout

```
heart_Risk_Prediction_Project/
├── backend/
│   ├── server.py                   # Production Flask REST API server (port 5000)
│   ├── clinical_copilot.py         # AI Clinical Copilot engine (Gemini / Groq / OpenAI)
│   ├── .env                        # API keys configuration (GEMINI_API_KEY, GROQ_API_KEY)
│   └── .env.example                # Example environment variables template
├── frontend/
│   ├── package.json                # React + Vite dependencies
│   ├── vite.config.js              # Vite server & reverse proxy to backend
│   ├── src/
│   │   ├── App.jsx                 # Clinical dashboard UI & state management
│   │   ├── styles.css              # Custom responsive glassmorphic styling
│   │   ├── main.jsx                # React entry point
│   │   ├── components/
│   │   │   └── SurvivalTrajectoryCurve.jsx  # Interactive Kaplan-Meier trajectory SVG component
│   │   └── data/
│   │       └── scoreLogic.js       # Fallback local scoring, presets & survival math
│   └── README.md                   # Frontend architecture notes
├── programs/
│   ├── 00_export_raw.sas           # SAS raw export
│   ├── 01_eda.py                   # Exploratory analysis
│   ├── 02_cleaning.py              # Data preprocessing
│   ├── 03_feature_engineering.py   # Clinical feature formulas
│   ├── 04_split.py                 # Stratified train/val/test
│   ├── 05_modeling.py              # Model training
│   ├── 06_evaluation.py            # Evaluation & calibration
│   ├── 07_survival.py              # Kaplan-Meier & Cox PH
│   ├── 08_explainability.py        # SHAP, odds ratios, importance
│   ├── 09_scorecard_fairness.py    # Scorecard & subgroup fairness
│   ├── 10_mlops_monitoring.py      # Drift monitoring (PSI)
│   ├── 11_scoring_deploy.py        # CLI deployment scorer
│   ├── run_all.py                  # Master pipeline runner
│   ├── config.py                   # Global path & random seed config
│   └── macros/
│       ├── score_patient.py        # Patient feature transformer & scorer
│       └── metrics_summary.py      # Metric logging helpers
├── data/                           # Raw & processed CSVs (gitignored)
├── models/                         # Serialized .pkl models (gitignored)
├── output/
│   ├── plots/                      # 12 high-resolution diagnostic plots
│   └── reports/                    # Odds ratios, scorecard points, fairness CSVs
├── monitoring/                     # PSI drift tables & experiment logs
├── requirements.txt                # Python dependencies
└── README.md                       # Master project documentation
```

---

## 🚀 Getting Started & Installation

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0 or higher (with `npm`)

### 1. Clone the Repository
```bash
git clone https://github.com/Purvasalunk1344/heart_Risk_Prediction_Project.git
cd heart_Risk_Prediction_Project
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 4. (Optional) Configure AI Copilot Keys
To enable Google Gemini or Groq AI care plan features, copy `backend/.env.example` to `backend/.env` and add your API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
# or
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🏃 How to Run the Project

### Option A: Run the End-to-End ML Pipeline
To retrain models and regenerate all plots and reports from scratch:
```powershell
cd programs
python run_all.py
```
*(Takes ~2.5 minutes to complete all 10 stages).*

---

### Option B: Run the Full-Stack Application (Backend + Frontend)

Open **two separate terminal windows**:

#### Terminal 1 — Start the Python API Backend:
```powershell
cd backend
python server.py
```
*Backend will start on `http://127.0.0.1:5000` with pre-cached models.*

#### Terminal 2 — Start the React Frontend:
```powershell
cd frontend
npm run dev
```
*Frontend will start on `http://localhost:5173`.*

👉 Open **[http://localhost:5173/](http://localhost:5173/)** in your browser.

---

## 📡 API Reference

### `GET /api/health`
Verifies backend connectivity and lists available models.
```json
{
  "status": "online",
  "service": "Heart Risk Prediction API",
  "champion_model": "logistic",
  "available_models": ["logistic", "random_forest", "gradient_boosting"],
  "threshold": 0.20
}
```

---

### `POST /api/predict`
Calculates risk, multi-model consensus, vitals diagnostics, and patient guidance.

**Request Body:**
```json
{
  "AgeAtStart": 58,
  "Sex": "Male",
  "Systolic": 165,
  "Diastolic": 96,
  "Cholesterol": 248,
  "Weight": 195,
  "Height": 68,
  "Smoking": 20,
  "model": "logistic"
}
```

**Response:**
```json
{
  "success": true,
  "model_used": "logistic",
  "risk": 0.9125,
  "risk_percentage": 91.3,
  "risk_band": "High",
  "flagged": true,
  "threshold": 0.2,
  "consensus": {
    "logistic": 0.9125,
    "random_forest": 0.9208,
    "gradient_boosting": 0.929
  },
  "clinical_metrics": {
    "bmi": 29.6,
    "bmi_status": "Overweight",
    "bp_status": "Stage 2 Hypertension",
    "chol_status": "High (>=240 mg/dL)",
    "pulse_pressure": 69,
    "smoke_status": "Moderate Smoker (20 cigs/day)"
  },
  "insights": {
    "urgency": "High Priority Clinical Alert",
    "summary": "Your calculated cardiovascular mortality risk profile is in the High tier (>50%). Prompt consultation with a healthcare provider is strongly recommended.",
    "elevating_factors": [
      "Age (58 yrs): Risk increases past age 50 due to vascular stiffening.",
      "Blood Pressure (165/96 mmHg - Stage 2 Hypertension): Elevates heart workload and mortality hazard.",
      "Smoking (Moderate Smoker (20 cigs/day)): Tobacco exposure damages vascular endothelium and more than doubles risk.",
      "Total Cholesterol (248 mg/dL): High serum lipids accelerate coronary plaque buildup."
    ],
    "protective_factors": [],
    "recommended_actions": [
      "Initiate or review antihypertensive therapy; aim for target BP under 130/80 mmHg.",
      "Enroll in a structured smoking cessation program.",
      "Perform full fasting lipid panel and discuss statin therapy with your physician."
    ]
  }
}
```

---

### `POST /api/survival-trajectory`
Calculates dynamic time-to-event Kaplan-Meier trajectories comparing untreated baseline vs. controlled intervention profiles at Years 1, 3, 5, 10, and 15.

**Request Body:**
```json
{
  "AgeAtStart": 58,
  "Sex": "Male",
  "Systolic": 165,
  "Diastolic": 96,
  "Cholesterol": 248,
  "Weight": 195,
  "Height": 68,
  "Smoking": 20,
  "model": "logistic"
}
```

**Response:**
```json
{
  "success": true,
  "model_used": "logistic",
  "baseline_10yr_risk": 91.3,
  "controlled_10yr_risk": 48.7,
  "absolute_reduction_10yr": 42.6,
  "trajectory": [
    { "year": 0, "untreated_risk": 0.0, "untreated_survival": 100.0, "controlled_risk": 0.0, "controlled_survival": 100.0, "absolute_risk_reduction": 0.0, "survival_gain": 0.0 },
    { "year": 1, "untreated_risk": 11.2, "untreated_survival": 88.8, "controlled_risk": 3.1, "controlled_survival": 96.9, "absolute_risk_reduction": 8.1, "survival_gain": 8.1 },
    { "year": 3, "untreated_risk": 39.5, "untreated_survival": 60.5, "controlled_risk": 12.8, "controlled_survival": 87.2, "absolute_risk_reduction": 26.7, "survival_gain": 26.7 },
    { "year": 5, "untreated_risk": 64.2, "untreated_survival": 35.8, "controlled_risk": 24.3, "controlled_survival": 75.7, "absolute_risk_reduction": 39.9, "survival_gain": 39.9 },
    { "year": 10, "untreated_risk": 91.3, "untreated_survival": 8.7, "controlled_risk": 48.7, "controlled_survival": 51.3, "absolute_risk_reduction": 42.6, "survival_gain": 42.6 },
    { "year": 15, "untreated_risk": 98.4, "untreated_survival": 1.6, "controlled_risk": 68.2, "controlled_survival": 31.8, "absolute_risk_reduction": 30.2, "survival_gain": 30.2 }
  ]
}
```

---

### `POST /api/copilot`
Interactive clinical AI conversation assistant for clinicians and patients.

**Request Body:**
```json
{
  "message": "What is the expected survival gain if the patient ceases tobacco usage?",
  "patient": {
    "AgeAtStart": 58,
    "Sex": "Male",
    "Systolic": 165,
    "Diastolic": 96,
    "Smoking": 20,
    "risk": 0.913
  },
  "history": []
}
```

---

### `POST /api/ai-guidance`
Generates a structured, clinical-grade 7-day care plan via Google Gemini AI.

---

### `GET /api/patients` & `POST /api/patients/save` & `DELETE /api/patients/<id>`
Manages saved patient records with instant JSON and CSV synchronization.

---

### `GET /api/presets`
Returns pre-configured clinical archetypes for demonstration.

---

### `GET /api/reports/summary`
Returns tabular statistical odds ratios and model card metrics.

---

## 📊 Model Performance & Evaluation

| Model | Validation ROC-AUC | Brier Calibration Score | Role |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.779** | **0.165** | 🏆 **Champion Model** (High interpretability, calibrated probabilities) |
| **Gradient Boosting** | 0.774 | 0.168 | Challenger (Captures non-linear feature interactions) |
| **Random Forest** | 0.768 | 0.171 | Challenger (Tree ensemble) |

- **Optimal Classification Threshold**: `0.20` (Calibrated to balance sensitivity for early life-saving clinical interventions).
- **Diagnostic Plots**: All 12 evaluation charts (ROC, Calibration, Kaplan-Meier, SHAP, Permutation Importance) are archived in [`output/plots/`](file:///c:/Data%20analytics/Project/heart_Risk_Prediction_Project/output/plots).

---

## ⚠️ Clinical Disclaimer

This software and its predictive models are designed for **research, educational, and clinical decision-support demonstration purposes only**. It does not constitute formal medical diagnosis or replace personalized judgment by a licensed medical practitioner. Any treatment, medication changes, or lifestyle protocols should always be evaluated in consultation with a qualified physician.
