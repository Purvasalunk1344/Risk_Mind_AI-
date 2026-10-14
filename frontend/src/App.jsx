import { useEffect, useState, useMemo, useRef } from 'react';
import { calculateRisk, getRiskBand, samplePatient } from './data/scoreLogic';

const fieldConfig = [
  { label: 'Age (Years)', key: 'AgeAtStart', type: 'number', min: 20, max: 100, step: 1 },
  { label: 'Sex', key: 'Sex', type: 'select', options: ['Female', 'Male'] },
  { label: 'Systolic Blood Pressure (mmHg)', key: 'Systolic', type: 'number', min: 80, max: 240, step: 1 },
  { label: 'Diastolic Blood Pressure (mmHg)', key: 'Diastolic', type: 'number', min: 40, max: 140, step: 1 },
  { label: 'Total Cholesterol (mg/dL)', key: 'Cholesterol', type: 'number', min: 100, max: 450, step: 1 },
  { label: 'Weight (lbs)', key: 'Weight', type: 'number', min: 80, max: 350, step: 1 },
  { label: 'Height (inches)', key: 'Height', type: 'number', min: 48, max: 84, step: 1 },
  { label: 'Smoking (cigarettes / day)', key: 'Smoking', type: 'number', min: 0, max: 80, step: 1 },
];

const PRESET_PATIENTS = [
  {
    id: 'high_risk',
    title: 'High Risk (Hypertensive Smoker)',
    data: {
      AgeAtStart: 58,
      Sex: 'Male',
      Systolic: 165,
      Diastolic: 96,
      Cholesterol: 248,
      Weight: 195,
      Height: 68,
      Smoking: 20,
    },
  },
  {
    id: 'moderate_risk',
    title: 'Moderate Risk (Borderline BP)',
    data: {
      AgeAtStart: 52,
      Sex: 'Female',
      Systolic: 138,
      Diastolic: 86,
      Cholesterol: 215,
      Weight: 155,
      Height: 65,
      Smoking: 5,
    },
  },
  {
    id: 'low_risk',
    title: 'Low Risk (Healthy Profile)',
    data: {
      AgeAtStart: 35,
      Sex: 'Female',
      Systolic: 115,
      Diastolic: 76,
      Cholesterol: 175,
      Weight: 130,
      Height: 65,
      Smoking: 0,
    },
  },
  {
    id: 'sample',
    title: 'Standard Framingham Sample',
    data: samplePatient,
  },
];

function App() {
  const [patient, setPatient] = useState(samplePatient);
  const [selectedModel, setSelectedModel] = useState('logistic');
  const [backendOnline, setBackendOnline] = useState(false);
  const [backendData, setBackendData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'factors' | 'guidance'
  const debounceTimer = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error('Backend not reachable');
        return res.json();
      })
      .then((data) => {
        if (data.status === 'online') {
          setBackendOnline(true);
        }
      })
      .catch(() => {
        setBackendOnline(false);
      });
  }, []);

  // Fetch prediction from connected backend when patient or model changes
  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);

    debounceTimer.current = setTimeout(() => {
      setLoading(true);
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...patient,
          model: selectedModel,
        }),
      })
        .then((res) => {
          if (!res.ok) throw new Error('Prediction API failed');
          return res.json();
        })
        .then((data) => {
          if (data.success) {
            setBackendOnline(true);
            setBackendData(data);
          }
          setLoading(false);
        })
        .catch(() => {
          setBackendOnline(false);
          setLoading(false);
        });
    }, 200);

    return () => clearTimeout(debounceTimer.current);
  }, [patient, selectedModel]);

  // Fallback local calculations if backend is loading or unavailable
  const localRisk = useMemo(() => calculateRisk(patient), [patient]);
  const localBand = useMemo(() => getRiskBand(localRisk), [localRisk]);

  const activeRisk = backendData?.risk ?? localRisk;
  const activeBand = backendData ? { label: backendData.risk_band, tone: backendData.risk_band.toLowerCase() } : localBand;
  const activePct = Math.round(activeRisk * 100);

  const updateField = (key, value) => {
    setPatient((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const loadPreset = (preset) => {
    setPatient(preset.data);
  };

  const clinical = backendData?.clinical_metrics || {
    bmi: patient.Weight && patient.Height ? ((patient.Weight * 703) / (patient.Height * patient.Height)).toFixed(1) : '24.8',
    bmi_status: 'Healthy',
    pulse_pressure: patient.Systolic && patient.Diastolic ? Number(patient.Systolic) - Number(patient.Diastolic) : 57,
    bp_status: Number(patient.Systolic) >= 140 ? 'Stage 2 Hypertension' : 'Normal',
    chol_status: Number(patient.Cholesterol) >= 240 ? 'High' : 'Desirable',
    smoke_status: Number(patient.Smoking) > 0 ? `${patient.Smoking} cigs/day` : 'Non-smoker',
  };

  const insights = backendData?.insights;

  return (
    <div className="app-shell">
      <div className="ambient ambient-1" />
      <div className="ambient ambient-2" />

      {/* TOPBAR */}
      <header className="topbar">
        <div className="brand-wrap">
          <div className="logo-dot" />
          <div>
            <p className="eyebrow">Clinical Decision Support System</p>
            <h1>HeartRisk Studio</h1>
          </div>
        </div>

        <div className="topbar-actions">
          <div className={`status-pill ${backendOnline ? 'online' : 'offline'}`}>
            <span className="dot" />
            <span>{backendOnline ? 'Python ML Backend (Port 5000)' : 'Local Fallback'}</span>
          </div>

          <div className="model-selector-wrap">
            <label htmlFor="model-select">ML Model:</label>
            <select
              id="model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="model-select"
            >
              <option value="logistic">Champion: Logistic Regression (AUC: 0.779)</option>
              <option value="random_forest">Random Forest Classifier</option>
              <option value="gradient_boosting">Gradient Boosting Classifier</option>
            </select>
          </div>
        </div>
      </header>

      {/* PRESETS BAR */}
      <section className="presets-section">
        <span className="presets-label">Quick Patient Presets:</span>
        <div className="presets-bar">
          {PRESET_PATIENTS.map((p) => (
            <button key={p.id} className="preset-chip" onClick={() => loadPreset(p)}>
              {p.title}
            </button>
          ))}
        </div>
      </section>

      <main className="dashboard">
        {/* HERO RESULTS PANEL */}
        <section className="hero-panel">
          <div className="hero-copy">
            <div className="badge-row">
              <span className="badge">Framingham Heart Study Cohort</span>
              <span className="badge model-badge">Active Model: {selectedModel.replace('_', ' ').toUpperCase()}</span>
            </div>

            <h2>Personalized Cardiovascular Mortality Risk</h2>
            <p>
              Real-time prognostic assessment derived from clinical risk factors, blood pressure hemodynamics, serum
              cholesterol, and tobacco exposure.
            </p>

            {/* Model Consensus Bar */}
            {backendData?.consensus && (
              <div className="consensus-card">
                <span className="consensus-title">Cross-Model Algorithm Consensus:</span>
                <div className="consensus-items">
                  <div className="consensus-item">
                    <span>Logistic:</span>
                    <strong>{Math.round(backendData.consensus.logistic * 100)}%</strong>
                  </div>
                  <div className="consensus-item">
                    <span>Random Forest:</span>
                    <strong>{Math.round(backendData.consensus.random_forest * 100)}%</strong>
                  </div>
                  <div className="consensus-item">
                    <span>Gradient Boosting:</span>
                    <strong>{Math.round(backendData.consensus.gradient_boosting * 100)}%</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* MAIN RISK CARD SHOWN TO PATIENT */}
          <div className={`risk-card ${activeBand.tone}`}>
            <div className="risk-header">
              <span className="label">10-Year Mortality Hazard</span>
              {loading && <span className="calculating-badge">Scoring...</span>}
            </div>

            <div className="risk-value">{activePct}%</div>
            
            <div className="risk-band-row">
              <span className={`risk-band-tag ${activeBand.tone}`}>{activeBand.label} Risk Tier</span>
              {backendData?.flagged && <span className="flag-alert">⚠️ High Risk Flag</span>}
            </div>

            <div className="risk-bar">
              <span style={{ width: `${Math.min(activeRisk * 100, 100)}%` }} />
            </div>

            <p className="risk-urgency-note">
              {insights?.urgency || (activeBand.label === 'High' ? 'High Priority Clinical Alert' : 'Routine Monitoring')}
            </p>
          </div>
        </section>

        {/* NAVIGATION TABS FOR PATIENT DISPLAY */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            📋 Patient Profile & Vitals
          </button>
          <button
            className={`tab-btn ${activeTab === 'factors' ? 'active' : ''}`}
            onClick={() => setActiveTab('factors')}
          >
            🔍 Why This Risk? (Contributing Factors)
          </button>
          <button
            className={`tab-btn ${activeTab === 'guidance' ? 'active' : ''}`}
            onClick={() => setActiveTab('guidance')}
          >
            💡 Patient Action Plan & Care Guidance
          </button>
        </div>

        {/* TAB 1: OVERVIEW & VITALS */}
        {activeTab === 'overview' && (
          <section className="content-grid">
            {/* INPUT PROFILE */}
            <div className="panel form-panel">
              <div className="panel-header">
                <h3>Patient Clinical Parameters</h3>
                <span>Input Biometrics</span>
              </div>

              <div className="field-grid">
                {fieldConfig.map((field) => (
                  <label key={field.key} className="field">
                    <span>{field.label}</span>
                    {field.type === 'select' ? (
                      <select
                        value={patient[field.key]}
                        onChange={(e) => updateField(field.key, e.target.value)}
                      >
                        {field.options.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={field.type}
                        min={field.min}
                        max={field.max}
                        step={field.step}
                        value={patient[field.key]}
                        onChange={(e) => updateField(field.key, e.target.value)}
                      />
                    )}
                  </label>
                ))}
              </div>
            </div>

            {/* LIVE VITALS INTERPRETATION SHOWN TO PATIENT */}
            <div className="panel insight-panel">
              <div className="panel-header">
                <h3>Vitals & Clinical Biomarkers</h3>
                <span>Patient Health Status</span>
              </div>

              <div className="stat-grid">
                <div className="stat-box">
                  <span>Blood Pressure Status</span>
                  <strong>{clinical.bp_status}</strong>
                  <small>{patient.Systolic}/{patient.Diastolic} mmHg</small>
                </div>

                <div className="stat-box">
                  <span>Pulse Pressure</span>
                  <strong>{clinical.pulse_pressure} mmHg</strong>
                  <small>Normal: 40-60 mmHg</small>
                </div>

                <div className="stat-box">
                  <span>Body Mass Index (BMI)</span>
                  <strong>{clinical.bmi}</strong>
                  <small>{clinical.bmi_status}</small>
                </div>

                <div className="stat-box">
                  <span>Lipid Profile</span>
                  <strong>{patient.Cholesterol} mg/dL</strong>
                  <small>{clinical.chol_status}</small>
                </div>

                <div className="stat-box full-width">
                  <span>Tobacco Exposure</span>
                  <strong>{clinical.smoke_status}</strong>
                  <small>{Number(patient.Smoking) === 0 ? 'Protective cardiovascular profile' : 'Elevates arterial plaque risk'}</small>
                </div>
              </div>

              <div className="summary-banner">
                <span className="summary-banner-title">Clinical Synthesis</span>
                <p>
                  {insights?.summary ||
                    'Prognostic probability is determined by multi-factor arterial hemodynamics and demographic interaction.'}
                </p>
              </div>
            </div>
          </section>
        )}

        {/* TAB 2: CONTRIBUTING FACTORS (EXPLAINABILITY) */}
        {activeTab === 'factors' && (
          <section className="factors-section">
            <div className="panel factors-panel">
              <div className="panel-header">
                <h3>What is Influencing the Patient's Heart Risk Score?</h3>
                <span>Model Explainability & Odds Ratios</span>
              </div>

              <p className="factors-intro">
                Our machine learning model explains which biomarkers and lifestyle factors are either elevating hazard
                or protecting the patient's arterial health.
              </p>

              <div className="factors-columns">
                {/* ELEVATING FACTORS */}
                <div className="factor-box danger">
                  <div className="factor-box-header">
                    <span className="icon">⚠️</span>
                    <h4>Primary Risk Elevators</h4>
                  </div>
                  <ul>
                    {insights?.elevating_factors && insights.elevating_factors.length > 0 ? (
                      insights.elevating_factors.map((factor, i) => (
                        <li key={i}>{factor}</li>
                      ))
                    ) : (
                      <li>No severe acute cardiovascular risk elevators detected for this profile.</li>
                    )}
                  </ul>
                </div>

                {/* PROTECTIVE FACTORS */}
                <div className="factor-box safe">
                  <div className="factor-box-header">
                    <span className="icon">🛡️</span>
                    <h4>Protective / Low-Risk Factors</h4>
                  </div>
                  <ul>
                    {insights?.protective_factors && insights.protective_factors.length > 0 ? (
                      insights.protective_factors.map((factor, i) => (
                        <li key={i}>{factor}</li>
                      ))
                    ) : (
                      <li>Current parameters warrant optimization across multiple categories.</li>
                    )}
                  </ul>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* TAB 3: PATIENT CARE PLAN & GUIDANCE */}
        {activeTab === 'guidance' && (
          <section className="guidance-section">
            <div className="panel guidance-panel">
              <div className="panel-header">
                <h3>Personalized Patient Care Plan & Action Items</h3>
                <span>Empathetic Clinical Guidance</span>
              </div>

              <div className="guidance-card-highlight">
                <div className="guidance-tier-tag">
                  Tier: <strong>{activeBand.label} Risk Priority</strong>
                </div>
                <p className="guidance-main-message">
                  {insights?.summary ||
                    'Patient should maintain communication with their primary healthcare provider for regular evaluations.'}
                </p>
              </div>

              <div className="action-steps-grid">
                <h4>Recommended Clinical Action Steps:</h4>
                <div className="steps-list">
                  {insights?.recommended_actions && insights.recommended_actions.length > 0 ? (
                    insights.recommended_actions.map((act, i) => (
                      <div className="step-card" key={i}>
                        <span className="step-num">{i + 1}</span>
                        <p>{act}</p>
                      </div>
                    ))
                  ) : (
                    <div className="step-card">
                      <span className="step-num">1</span>
                      <p>Continue routine wellness checkups and balanced nutrition.</p>
                    </div>
                  )}
                </div>

                <div className="patient-targets-box">
                  <h4>Key Patient Target Milestones:</h4>
                  <div className="milestones-row">
                    <div className="milestone">
                      <span className="milestone-label">Target Blood Pressure</span>
                      <strong>&lt; 120/80 mmHg</strong>
                    </div>
                    <div className="milestone">
                      <span className="milestone-label">Target Total Cholesterol</span>
                      <strong>&lt; 200 mg/dL</strong>
                    </div>
                    <div className="milestone">
                      <span className="milestone-label">Target Tobacco Use</span>
                      <strong>0 cigs / day</strong>
                    </div>
                    <div className="milestone">
                      <span className="milestone-label">Target Aerobic Activity</span>
                      <strong>150 min / week</strong>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
