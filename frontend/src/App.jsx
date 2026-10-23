import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { calculateRisk, getRiskBand, samplePatient } from './data/scoreLogic';

const fieldConfig = [
  { label: 'Age (Years)', key: 'AgeAtStart', type: 'number', min: 20, max: 100, step: 1, hint: 'Patient age between 20-100' },
  { label: 'Sex', key: 'Sex', type: 'select', options: ['Female', 'Male'], hint: 'Biological sex' },
  { label: 'Systolic Blood Pressure (mmHg)', key: 'Systolic', type: 'number', min: 80, max: 240, step: 1, hint: 'Target: < 120 mmHg' },
  { label: 'Diastolic Blood Pressure (mmHg)', key: 'Diastolic', type: 'number', min: 40, max: 140, step: 1, hint: 'Target: < 80 mmHg' },
  { label: 'Total Cholesterol (mg/dL)', key: 'Cholesterol', type: 'number', min: 100, max: 450, step: 1, hint: 'Desirable: < 200 mg/dL' },
  { label: 'Weight (lbs)', key: 'Weight', type: 'number', min: 80, max: 350, step: 1, hint: 'Body weight in pounds' },
  { label: 'Height (inches)', key: 'Height', type: 'number', min: 48, max: 84, step: 1, hint: 'Height (e.g. 66 in = 5ft 6in)' },
  { label: 'Smoking (cigarettes / day)', key: 'Smoking', type: 'number', min: 0, max: 80, step: 1, hint: '0 for non-smokers' },
];

const DEFAULT_PATIENT = {
  name: 'Framingham Sample Subject #1042',
  AgeAtStart: 58,
  Sex: 'Female',
  Systolic: 145,
  Diastolic: 88,
  Cholesterol: 210,
  Weight: 160,
  Height: 66,
  Smoking: 12,
  notes: 'Subject exhibits mild Stage 1 hypertension and moderate tobacco usage.',
};

const BLANK_PATIENT = {
  name: '',
  AgeAtStart: '',
  Sex: 'Female',
  Systolic: '',
  Diastolic: '',
  Cholesterol: '',
  Weight: '',
  Height: '',
  Smoking: 0,
  notes: '',
};

const PRESET_PATIENTS = [
  {
    id: 'high_risk',
    title: 'High Risk (Hypertensive Smoker)',
    data: {
      name: 'Patient - High Risk Profile',
      AgeAtStart: 58,
      Sex: 'Male',
      Systolic: 165,
      Diastolic: 96,
      Cholesterol: 248,
      Weight: 195,
      Height: 68,
      Smoking: 20,
      notes: 'Severe Stage 2 hypertension with regular tobacco exposure.',
    },
  },
  {
    id: 'moderate_risk',
    title: 'Moderate Risk (Borderline BP)',
    data: {
      name: 'Patient - Moderate Risk Profile',
      AgeAtStart: 52,
      Sex: 'Female',
      Systolic: 138,
      Diastolic: 86,
      Cholesterol: 215,
      Weight: 155,
      Height: 65,
      Smoking: 5,
      notes: 'Borderline elevated systolic blood pressure and cholesterol.',
    },
  },
  {
    id: 'low_risk',
    title: 'Low Risk (Healthy Profile)',
    data: {
      name: 'Patient - Low Risk Healthy Profile',
      AgeAtStart: 35,
      Sex: 'Female',
      Systolic: 115,
      Diastolic: 76,
      Cholesterol: 175,
      Weight: 130,
      Height: 65,
      Smoking: 0,
      notes: 'Non-smoker with optimal blood pressure and healthy BMI.',
    },
  },
  {
    id: 'sample',
    title: 'Standard Framingham Sample',
    data: DEFAULT_PATIENT,
  },
];

function App() {
  const [patient, setPatient] = useState(DEFAULT_PATIENT);
  const [selectedModel, setSelectedModel] = useState('logistic');
  const [backendOnline, setBackendOnline] = useState(false);
  const [backendData, setBackendData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [savingRecord, setSavingRecord] = useState(false);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'factors' | 'guidance' | 'history'
  const [hasUncalculatedChanges, setHasUncalculatedChanges] = useState(false);
  const [savedPatients, setSavedPatients] = useState([]);
  const [historySearch, setHistorySearch] = useState('');
  const [toast, setToast] = useState(null);
  const toastTimeoutRef = useRef(null);

  // Trigger floating toast message
  const showToast = useCallback((message, type = 'success') => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setToast({ message, type });
    toastTimeoutRef.current = setTimeout(() => {
      setToast(null);
    }, 3500);
  }, []);

  // Fetch saved patient records from backend or localStorage
  const loadSavedHistory = useCallback(() => {
    fetch('/api/patients')
      .then((res) => {
        if (!res.ok) throw new Error('API offline');
        return res.json();
      })
      .then((data) => {
        if (data.success && Array.isArray(data.patients)) {
          setSavedPatients(data.patients);
          localStorage.setItem('heart_saved_patients', JSON.stringify(data.patients));
        }
      })
      .catch(() => {
        // Fallback to local storage
        try {
          const stored = localStorage.getItem('heart_saved_patients');
          if (stored) {
            setSavedPatients(JSON.parse(stored));
          }
        } catch (e) {
          console.warn('Could not read saved patients from localStorage', e);
        }
      });
  }, []);

  // Check backend health on mount and load history
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

    loadSavedHistory();
  }, [loadSavedHistory]);

  // Unified function to execute ML prediction
  const executePrediction = useCallback(
    (currentPatient, modelName) => {
      setLoading(true);
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...currentPatient,
          model: modelName,
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
            setHasUncalculatedChanges(false);
          }
          setLoading(false);
        })
        .catch(() => {
          setBackendOnline(false);
          setLoading(false);
          setHasUncalculatedChanges(false);
        });
    },
    []
  );

  // Initial calculation on page load
  useEffect(() => {
    executePrediction(DEFAULT_PATIENT, selectedModel);
  }, [executePrediction, selectedModel]);

  // Model switch re-scores automatically
  const handleModelChange = (newModel) => {
    setSelectedModel(newModel);
    executePrediction(patient, newModel);
    showToast(`Switched active model to: ${newModel.replace('_', ' ').toUpperCase()}`, 'info');
  };

  // Fallback local calculations if backend is loading or unavailable
  const localRisk = useMemo(() => calculateRisk(patient), [patient]);
  const localBand = useMemo(() => getRiskBand(localRisk), [localRisk]);

  const activeRisk = backendData?.risk ?? localRisk;
  const activeBand = backendData ? { label: backendData.risk_band, tone: backendData.risk_band.toLowerCase() } : localBand;
  const activePct = Math.round(activeRisk * 100);

  // Update field value and mark uncalculated changes
  const updateField = (key, value) => {
    setPatient((current) => ({
      ...current,
      [key]: value,
    }));
    setHasUncalculatedChanges(true);
  };

  // Explicit calculation button click
  const handleCalculateClick = () => {
    executePrediction(patient, selectedModel);
    showToast('Heart risk score calculated successfully!');
  };

  // Load a preset patient profile
  const loadPreset = (preset) => {
    setPatient(preset.data);
    executePrediction(preset.data, selectedModel);
    showToast(`Loaded preset: ${preset.title}`);
  };

  // Clear form to type brand new data
  const handleClearForm = () => {
    setPatient(BLANK_PATIENT);
    setHasUncalculatedChanges(true);
    showToast('Form cleared. Please enter new patient biometrics and click "Calculate Heart Risk".', 'info');
  };

  // Reset to default sample
  const handleResetDefault = () => {
    setPatient(DEFAULT_PATIENT);
    executePrediction(DEFAULT_PATIENT, selectedModel);
    showToast('Reset form to standard Framingham reference patient.');
  };

  // Save current patient record to backend and dataset history
  const handleSavePatient = () => {
    const recordName = (patient.name || '').trim() || `Patient #${Math.floor(1000 + Math.random() * 9000)}`;
    setSavingRecord(true);

    const payload = {
      ...patient,
      name: recordName,
      model: selectedModel,
      risk: activeRisk,
      risk_percentage: activePct,
      risk_band: activeBand.label,
    };

    fetch('/api/patients/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to save to server');
        return res.json();
      })
      .then((data) => {
        setSavingRecord(false);
        if (data.success && data.patient) {
          const updated = [data.patient, ...savedPatients.filter((p) => p.id !== data.patient.id)];
          setSavedPatients(updated);
          localStorage.setItem('heart_saved_patients', JSON.stringify(updated));
          showToast(`✅ Patient "${recordName}" saved to history and dataset!`);
        }
      })
      .catch(() => {
        // Local fallback save if server is unreachable
        const localRecord = {
          id: `pat_local_${Date.now()}`,
          name: recordName,
          created_at: new Date().toISOString().replace('T', ' ').slice(0, 19),
          ...payload,
        };
        const updated = [localRecord, ...savedPatients];
        setSavedPatients(updated);
        localStorage.setItem('heart_saved_patients', JSON.stringify(updated));
        setSavingRecord(false);
        showToast(`✅ Patient "${recordName}" saved locally!`);
      });
  };

  // Load a previously saved patient back into the calculator
  const handleLoadSavedPatient = (record) => {
    const loadedData = {
      name: record.name || '',
      AgeAtStart: record.AgeAtStart || 50,
      Sex: record.Sex || 'Female',
      Systolic: record.Systolic || 120,
      Diastolic: record.Diastolic || 80,
      Cholesterol: record.Cholesterol || 200,
      Weight: record.Weight || 160,
      Height: record.Height || 66,
      Smoking: record.Smoking ?? 0,
      notes: record.notes || '',
    };
    setPatient(loadedData);
    if (record.model_used) {
      setSelectedModel(record.model_used);
    }
    executePrediction(loadedData, record.model_used || selectedModel);
    setActiveTab('overview');
    showToast(`Loaded "${record.name}" into clinical calculator!`);
  };

  // Delete a saved patient
  const handleDeleteSavedPatient = (patientId, patientName) => {
    if (!window.confirm(`Delete saved record for "${patientName || 'this patient'}"?`)) {
      return;
    }
    fetch(`/api/patients/${patientId}`, { method: 'DELETE' })
      .catch(() => {})
      .finally(() => {
        const updated = savedPatients.filter((p) => p.id !== patientId);
        setSavedPatients(updated);
        localStorage.setItem('heart_saved_patients', JSON.stringify(updated));
        showToast(`Deleted record "${patientName}".`, 'info');
      });
  };

  // Export saved patients as JSON
  const handleExportJSON = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(savedPatients, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `saved_heart_risk_patients_${new Date().toISOString().slice(0, 10)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showToast('Exported patient records JSON file.');
  };

  // Clinical metrics for interpretation panel
  const clinical = backendData?.clinical_metrics || {
    bmi: patient.Weight && patient.Height ? ((patient.Weight * 703) / (patient.Height * patient.Height)).toFixed(1) : '24.8',
    bmi_status: 'Healthy',
    pulse_pressure: patient.Systolic && patient.Diastolic ? Number(patient.Systolic) - Number(patient.Diastolic) : 57,
    bp_status: Number(patient.Systolic) >= 140 ? 'Stage 2 Hypertension' : 'Normal',
    chol_status: Number(patient.Cholesterol) >= 240 ? 'High' : 'Desirable',
    smoke_status: Number(patient.Smoking) > 0 ? `${patient.Smoking} cigs/day` : 'Non-smoker',
  };

  const insights = backendData?.insights;

  // Filter saved records for Tab 4
  const filteredSavedPatients = useMemo(() => {
    if (!historySearch.trim()) return savedPatients;
    const q = historySearch.toLowerCase();
    return savedPatients.filter(
      (p) =>
        (p.name && p.name.toLowerCase().includes(q)) ||
        (p.risk_band && p.risk_band.toLowerCase().includes(q)) ||
        (p.Sex && p.Sex.toLowerCase().includes(q))
    );
  }, [savedPatients, historySearch]);

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

          {/* MAIN RISK CARD */}
          <div className={`risk-card ${activeBand.tone}`}>
            <div className="risk-header">
              <span className="label">10-Year Mortality Hazard</span>
              {loading && <span className="calculating-badge">Scoring Patient...</span>}
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

            {/* Quick Save button in Hero Card */}
            <div className="hero-quick-save">
              <button
                className="btn-save-record"
                onClick={handleSavePatient}
                disabled={savingRecord}
                title="Save current evaluated patient into permanent records"
              >
                💾 {savingRecord ? 'Saving...' : 'Save Patient Record'}
              </button>
            </div>
          </div>
        </section>

        {/* NAVIGATION TABS */}
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
          <button
            className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            📂 Saved Patients & History
            <span className="badge-counter">{savedPatients.length}</span>
          </button>
        </div>

        {/* TAB 1: OVERVIEW & VITALS (FORM + PREDICTION TRIGGER) */}
        {activeTab === 'overview' && (
          <section className="content-grid">
            {/* INPUT PROFILE PANEL */}
            <div className="panel form-panel">
              <div className="panel-header">
                <div>
                  <h3>Patient Clinical Parameters</h3>
                  <span>Input Biometrics & Enter New Data</span>
                </div>
              </div>

              <div className="field-grid">
                {/* PATIENT NAME / IDENTIFIER */}
                <label className="field full-field">
                  <span>Patient Name / Medical Record ID</span>
                  <input
                    type="text"
                    placeholder="e.g. Eleanor Vance or Patient #104"
                    value={patient.name || ''}
                    onChange={(e) => updateField('name', e.target.value)}
                  />
                </label>

                {/* BIOMETRIC INPUTS */}
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
                        placeholder={field.hint}
                        value={patient[field.key] ?? ''}
                        onChange={(e) => updateField(field.key, e.target.value)}
                      />
                    )}
                  </label>
                ))}

                {/* CLINICAL NOTES */}
                <label className="field full-field">
                  <span>Clinical Notes & Observations (Optional)</span>
                  <textarea
                    placeholder="e.g. Family history of heart disease, taking beta-blockers..."
                    value={patient.notes || ''}
                    onChange={(e) => updateField('notes', e.target.value)}
                    rows={2}
                  />
                </label>
              </div>

              {/* ACTION TOOLBAR: EXPLICIT PREDICT, SAVE, CLEAR, RESET */}
              <div className="form-actions-toolbar">
                <div className="form-status-bar">
                  <span className={`status-text ${hasUncalculatedChanges ? 'modified' : 'synced'}`}>
                    {hasUncalculatedChanges
                      ? '⚠️ Parameters modified — click "Calculate Heart Risk" to score'
                      : '✅ Prediction is up to date with active parameters'}
                  </span>
                  {patient.name && <small style={{ color: '#8db4e2' }}>Active: {patient.name}</small>}
                </div>

                <div className="buttons-row">
                  {/* CHANGE 1: EXPLICIT CALCULATE BUTTON */}
                  <button
                    className="btn-primary-action"
                    onClick={handleCalculateClick}
                    disabled={loading}
                  >
                    <span>⚡</span>
                    <span>{loading ? 'Evaluating Model...' : 'Calculate Heart Risk'}</span>
                  </button>

                  {/* CHANGE 3: SAVE PATIENT RECORD BUTTON */}
                  <button
                    className="btn-save-record"
                    onClick={handleSavePatient}
                    disabled={savingRecord}
                  >
                    <span>💾</span>
                    <span>{savingRecord ? 'Saving...' : 'Save Patient Record'}</span>
                  </button>

                  {/* CHANGE 1: CLEAR / RESET BUTTONS */}
                  <button
                    className="btn-secondary-action"
                    onClick={handleClearForm}
                    title="Clear all fields to type a brand new patient"
                  >
                    🧹 Clear Form
                  </button>

                  <button
                    className="btn-secondary-action"
                    onClick={handleResetDefault}
                    title="Reset to default Framingham reference sample"
                  >
                    🔄 Reset Sample
                  </button>
                </div>
              </div>
            </div>

            {/* LIVE VITALS INTERPRETATION */}
            <div className="panel insight-panel">
              <div className="panel-header">
                <div>
                  <h3>Vitals & Clinical Biomarkers</h3>
                  <span>Patient Health Status</span>
                </div>
              </div>

              <div className="stat-grid">
                <div className="stat-box">
                  <span>Blood Pressure Status</span>
                  <strong>{clinical.bp_status}</strong>
                  <small>{patient.Systolic || '—'}/{patient.Diastolic || '—'} mmHg</small>
                </div>

                <div className="stat-box">
                  <span>Pulse Pressure</span>
                  <strong>{clinical.pulse_pressure} mmHg</strong>
                  <small>Normal range: 40-60 mmHg</small>
                </div>

                <div className="stat-box">
                  <span>Body Mass Index (BMI)</span>
                  <strong>{clinical.bmi}</strong>
                  <small>{clinical.bmi_status}</small>
                </div>

                <div className="stat-box">
                  <span>Lipid Profile</span>
                  <strong>{patient.Cholesterol || '—'} mg/dL</strong>
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
                <div>
                  <h3>What is Influencing the Patient's Heart Risk Score?</h3>
                  <span>Model Explainability & Odds Ratios</span>
                </div>
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
                <div>
                  <h3>Personalized Patient Care Plan & Action Items</h3>
                  <span>Empathetic Clinical Guidance</span>
                </div>
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

        {/* TAB 4: SAVED PATIENTS & HISTORY */}
        {activeTab === 'history' && (
          <section className="saved-patients-section">
            <div className="history-panel">
              <div className="panel-header">
                <div>
                  <h3>Saved Patient Records & Cohort History</h3>
                  <span>Longitudinal Patient Directory & Dataset Entries</span>
                </div>
                <button className="btn-secondary-action" onClick={handleExportJSON} disabled={savedPatients.length === 0}>
                  📥 Export Records (JSON)
                </button>
              </div>

              <div className="history-toolbar">
                <input
                  type="text"
                  className="history-search-input"
                  placeholder="🔍 Search patients by name, sex, or risk tier..."
                  value={historySearch}
                  onChange={(e) => setHistorySearch(e.target.value)}
                />
                <span style={{ fontSize: '0.85rem', color: '#9bb8dc' }}>
                  Showing {filteredSavedPatients.length} of {savedPatients.length} saved records
                </span>
              </div>

              {filteredSavedPatients.length === 0 ? (
                <div className="empty-history-card">
                  <div className="empty-history-icon">📋</div>
                  <h4>No Saved Patient Records Found</h4>
                  <p style={{ color: '#8ea8c8', maxWidth: '420px', margin: '8px auto 16px' }}>
                    {savedPatients.length === 0
                      ? 'You have not saved any patients yet. Go to the "Patient Profile & Vitals" tab, enter a patient\'s data, and click "Save Patient Record".'
                      : 'No records match your search query.'}
                  </p>
                  <button className="btn-primary-action" style={{ maxWidth: '220px', margin: '0 auto' }} onClick={() => setActiveTab('overview')}>
                    Go to Patient Calculator
                  </button>
                </div>
              ) : (
                <div className="table-responsive">
                  <table className="records-table">
                    <thead>
                      <tr>
                        <th>Patient Name & ID</th>
                        <th>Saved On</th>
                        <th>Age / Sex</th>
                        <th>Blood Pressure</th>
                        <th>Cholesterol</th>
                        <th>BMI</th>
                        <th>Smoking</th>
                        <th>Model</th>
                        <th>Heart Risk</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredSavedPatients.map((rec) => (
                        <tr key={rec.id}>
                          <td>
                            <span className="patient-cell-name">{rec.name || 'Unnamed Patient'}</span>
                            <span className="patient-cell-id">{rec.id}</span>
                          </td>
                          <td style={{ whiteSpace: 'nowrap', color: '#9cb5d6' }}>
                            {rec.created_at || 'Just now'}
                          </td>
                          <td>
                            {rec.AgeAtStart} yrs • {rec.Sex}
                          </td>
                          <td>
                            <strong>{rec.Systolic}/{rec.Diastolic}</strong> mmHg
                          </td>
                          <td>
                            {rec.Cholesterol} mg/dL
                          </td>
                          <td>
                            {rec.BMI || ((rec.Weight * 703) / (rec.Height * rec.Height)).toFixed(1)}
                          </td>
                          <td>
                            {Number(rec.Smoking) > 0 ? `${rec.Smoking} cigs/day` : 'Non-smoker'}
                          </td>
                          <td style={{ textTransform: 'capitalize', color: '#88b5ec' }}>
                            {rec.model_used || 'logistic'}
                          </td>
                          <td>
                            <span className={`risk-band-tag ${(rec.risk_band || 'low').toLowerCase()}`}>
                              {rec.risk_percentage ?? Math.round((rec.risk || 0.1) * 100)}% ({rec.risk_band || 'Low'})
                            </span>
                          </td>
                          <td>
                            <div className="table-actions">
                              <button
                                className="btn-action-load"
                                onClick={() => handleLoadSavedPatient(rec)}
                                title="Load this patient into the form calculator"
                              >
                                📥 Load
                              </button>
                              <button
                                className="btn-action-delete"
                                onClick={() => handleDeleteSavedPatient(rec.id, rec.name)}
                                title="Delete record"
                              >
                                🗑️
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      {/* FLOATING TOAST FEEDBACK */}
      {toast && (
        <div className="toast-banner">
          <span>{toast.message}</span>
        </div>
      )}
    </div>
  );
}

export default App;
