# ❤️ Heart Disease Mortality Risk Prediction System & Clinical Studio

An end-to-end clinical machine learning and decision-support system trained on the **Framingham Heart Study** cohort (`SASHELP.HEART`). The system predicts **10-year cardiovascular mortality risk**, explains patient-level risk drivers, evaluates multi-model algorithm consensus, monitors data drift, and provides an interactive clinical web dashboard for healthcare providers and patients.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [Key Features](#-key-features)
- [Full-Stack System Architecture](#-full-stack-system-architecture)
- [What the Frontend Tells the User](#-what-the-frontend-tells-the-user)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Project Directory Layout](#-project-directory-layout)
- [Getting Started & Installation](#-getting-started--installation)
- [How to Run the Project](#-how-to-run-the-project)
- [API Reference](#-api-reference)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Clinical Disclaimer](#-clinical-disclaimer)

---

## 🩺 Executive Summary

Cardiovascular disease remains the leading cause of global mortality. Early, accurate prognostic assessment enables proactive interventions that prevent fatal cardiac events. 

This repository delivers:
1. **A 10-Stage Machine Learning Pipeline**: Covering exploratory data analysis, data cleaning, hemodynamic feature engineering, train/val/test stratified splitting, model training, survival analysis (Kaplan-Meier & Cox PH), explainable AI (SHAP & Odds Ratios), scorecard creation, fairness auditing, and MLOps drift monitoring.
2. **A Python REST API Backend (`Flask`)**: Serves real-time inference, preloads trained models, calculates clinical metrics, and formats explainability insights.
3. **An Interactive Clinical Web Studio (`React + Vite`)**: A dark-mode, glassmorphic decision-support application where clinicians and patients explore risk profiles, view cross-model consensus, review physiological contributing factors, and receive customized care plans.

---

## ✨ Key Features

- **Multi-Model Intelligence**: Simultaneously evaluates patients against three distinct machine learning algorithms:
  - 🏆 **Champion**: Logistic Regression with standardized scaling (Validation AUC: `0.779`)
  - **Random Forest Classifier**: Ensemble decision tree model
  - **Gradient Boosting Classifier**: Sequentially optimized gradient trees
- **Survival Analysis**: Computes time-to-event survival trajectories using Kaplan-Meier survival curves and Cox Proportional Hazards modeling.
- **Explainable AI (XAI)**:
  - Statistical **Odds Ratios** with 95% Confidence Intervals via `statsmodels`
  - **SHAP (SHapley Additive exPlanations)** values and summary plots
  - **Permutation Feature Importance**
- **Fairness & MLOps Governance**:
  - Subgroup fairness checks across biological sex and age demographics
  - Population Stability Index (PSI) drift monitoring for production deployments
- **Interactive Patient Presets**: Instant 1-click loading of diverse clinical archetypes (High-Risk Hypertensive Smoker, Moderate Risk, Low Risk / Healthy, Senior Profile).

---

## 🏛️ Full-Stack System Architecture

```
+-----------------------------------------------------------------------------------+
|                            FRAMINGHAM COHORT DATA                                 |
|                             (SASHELP.HEART)                                       |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         10-STAGE PYTHON ML PIPELINE                               |
|   01_eda -> 02_cleaning -> 03_feature_engineering -> 04_split -> 05_modeling     |
|   -> 06_evaluation -> 07_survival -> 08_explainability -> 09_fairness -> 10_mlops |
+-----------------------------------------+-----------------------------------------+
                                          | Generates
                                          v
+-----------------------------------------------------------------------------------+
|                         MODELS & ANALYTICS ARTIFACTS                              |
|   models/logistic.pkl | models/random_forest.pkl | models/gradient_boosting.pkl   |
|   output/plots/*.png  | output/reports/*.csv     | monitoring/drift_summary.json  |
+-----------------------------------------+-----------------------------------------+
                                          | Preloaded by
                                          v
+-----------------------------------------------------------------------------------+
|                         FLASK REST API BACKEND (Port 5000)                        |
|   - Pre-cached Scikit-Learn Pipelines       - Derived Clinical Biomarker Math     |
|   - Multi-Model Consensus Scoring           - Plain-Language Explainability       |
+-----------------------------------------+-----------------------------------------+
                                          ^ Reverse Proxy (/api/*)
                                          |
+-----------------------------------------+-----------------------------------------+
|                      REACT + VITE CLINICAL STUDIO (Port 5173)                     |
|   - Live Mortality Risk Gauge               - Vitals & Hemodynamics Diagnosis     |
|   - Cross-Model Consensus Bar               - "Why This Risk?" Contributing List  |
|   - 1-Click Patient Archetype Presets       - Actionable Patient Care Plan        |
+-----------------------------------------------------------------------------------+
```

---

## 🖥️ What the Frontend Tells the User

The frontend acts as an empathetic **digital cardiology consultant**, translating raw medical metrics into four clinical answers:

```
+-----------------------------------------------------------------------------------------+
|  HeartRisk Studio                  [ 🟢 Python ML Backend (Port 5000) ]  [ Model Selector ] |
|  [ Quick Presets: High Risk | Moderate Risk | Healthy / Low Risk | Senior Profile ]        |
+-----------------------------------------------------------------------------------------+
|  HERO RESULTS PANEL:                                                                    |
|  - 10-Year Mortality Hazard Gauge: e.g., 91% (HIGH RISK TIER)                            |
|  - Cross-Model Consensus: Logistic (91%) | Random Forest (92%) | Gradient Boosting (93%) |
+-----------------------------------------------------------------------------------------+
|  [ 📋 Patient Profile & Vitals ]  [ 🔍 Why This Risk? ]  [ 💡 Patient Action Plan ]     |
+-----------------------------------------------------------------------------------------+
```

### 1. "How safe or at-risk is my heart?" (The Risk Verdict)
- **10-Year Mortality Hazard**: Clear percentage gauge (e.g., **`91%`**).
- **Risk Tier Badge**:
  - 🟢 **Low Risk (< 20%)**: Heart profile is within safe parameters; routine maintenance.
  - 🟡 **Medium Risk (20% – 50%)**: Borderline warning; proactive lifestyle/BP management needed.
  - 🔴 **High Risk (> 50%)**: Clinical alert; requires prompt cardiology review.
- **Algorithm Consensus**: Displays predictions from Logistic Regression, Random Forest, and Gradient Boosting side-by-side to ensure diagnostic consensus.

### 2. "How healthy are my individual body measurements?" (Vitals Panel)
- **Blood Pressure Categorization**: Classifies readings into *Optimal*, *Elevated*, *Stage 1 Hypertension*, or *Stage 2 Hypertension* (e.g. `165/96 mmHg`).
- **Pulse Pressure**: Calculates arterial stiffness ($Systolic - Diastolic$, normal 40–60 mmHg).
- **Body Mass Index (BMI)**: Computes BMI from height and weight ($BMI = \frac{Weight \times 703}{Height^2}$) with clinical categorization (*Healthy*, *Overweight*, or *Obese*).
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

### 4. "What should I do right now to improve my health?" (Action Plan)
- **Urgency Notice**: Clear guidance based on tier (urgent doctor consultation vs. routine annual screening).
- **Prescriptive Action Steps**:
  1. *Initiate or review antihypertensive therapy; aim for target BP under 130/80 mmHg.*
  2. *Enroll in a structured smoking cessation program (nicotine replacement therapy, counseling).*
  3. *Order a full fasting lipid panel and discuss statin therapy with a physician.*
- **Concrete Patient Target Milestones**:
  - Target Blood Pressure: **< 120/80 mmHg**
  - Target Cholesterol: **< 200 mg/dL**
  - Target Tobacco Use: **0 cigarettes/day**
  - Target Physical Activity: **150 min/week aerobic exercise**

---

## 🔬 Machine Learning Pipeline

The pipeline is organized into sequential, modular Python scripts located in [`programs/`](file:///c:/Users/sneha/OneDrive/Desktop/Heart_Risk_Project/heart_Risk_Prediction_Project/programs):

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
│   └── server.py                   # Production Flask REST API server (port 5000)
├── frontend/
│   ├── package.json                # React + Vite dependencies
│   ├── vite.config.js              # Vite server & reverse proxy to backend
│   ├── src/
│   │   ├── App.jsx                 # Clinical dashboard UI & state management
│   │   ├── styles.css              # Custom responsive glassmorphic styling
│   │   ├── main.jsx                # React entry point
│   │   └── data/
│   │       └── scoreLogic.js       # Fallback local calculations & presets
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

---

## 🏃 How to Run the Project

### Option A: Run the End-to-End ML Pipeline
To retrain models and regenerate all plots and reports from scratch:
```bash
cd programs
python run_all.py
```
*(Takes ~2.5 minutes to complete all 10 stages).*

### Option B: Run the Full-Stack Application (Backend + Frontend)

Open two terminal windows:

#### Terminal 1 — Start the Python API Backend:
```bash
python backend/server.py
```
*Backend will start on `http://127.0.0.1:5000` with pre-cached models.*

#### Terminal 2 — Start the React Frontend:
```bash
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

### `GET /api/presets`
Returns pre-configured clinical archetypes for demonstration.

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
- **Diagnostic Plots**: All 12 evaluation charts (ROC, Calibration, Kaplan-Meier, SHAP, Permutation Importance) are archived in [`output/plots/`](file:///c:/Users/sneha/OneDrive/Desktop/Heart_Risk_Project/heart_Risk_Prediction_Project/output/plots).

---

## ⚠️ Clinical Disclaimer

This software and its predictive models are designed for **research, educational, and clinical decision-support demonstration purposes only**. It does not constitute formal medical diagnosis or replace personalized judgment by a licensed medical practitioner. Any treatment, medication changes, or lifestyle protocols should always be evaluated in consultation with a qualified physician.
