import { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { getRiskBand, samplePatient, generateCounterfactualPlan } from './data/scoreLogic';
import SurvivalTrajectoryCurve from './components/SurvivalTrajectoryCurve';

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
  const [survivalData, setSurvivalData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [savingRecord, setSavingRecord] = useState(false);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'factors' | 'guidance' | 'history'
  const [hasUncalculatedChanges, setHasUncalculatedChanges] = useState(false);
  const [savedPatients, setSavedPatients] = useState([]);
  const [historySearch, setHistorySearch] = useState('');
  const [targetRiskGoal, setTargetRiskGoal] = useState(20);
  const [toast, setToast] = useState(null);
  const toastTimeoutRef = useRef(null);
  const [aiGuidance, setAiGuidance] = useState(null);
  const [loadingAI, setLoadingAI] = useState(false);
  const [aiError, setAiError] = useState(null);

  // ── CLINICAL AI COPILOT STATE ──────────────────────────────────────────
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([
    {
      role: 'assistant',
      content: '👋 **Welcome, Doctor.**\n\nI\'m your Clinical AI Copilot — ask me anything about this patient\'s cardiovascular profile, pharmacotherapy, dietary interventions, or interpretation of flagged metrics.\n\n**Example questions:**\n- Why was pulse pressure flagged in yellow?\n- What dietary changes lower systolic pressure fastest?\n- What medications should I consider for this patient?',
    },
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef(null);
  const chatInputRef = useRef(null);

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
          if (!res.ok) throw new Error('Model service unavailable');
          return res.json();
        })
        .then((data) => {
          if (data.success) {
            setBackendOnline(true);
            setBackendData(data);
            setHasUncalculatedChanges(false);
          } else {
            setBackendOnline(false);
            setBackendData(null);
          }
          setLoading(false);
        })
        .catch(() => {
          setBackendOnline(false);
          setBackendData(null);
          setLoading(false);
          setHasUncalculatedChanges(false);
        });

      // Also execute survival curve prediction
      fetch('/api/survival', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentPatient),
      })
        .then((res) => {
          if (!res.ok) throw new Error('Survival API unavailable');
          return res.json();
        })
        .then((data) => {
          if (data.success) {
            setSurvivalData(data);
          }
        })
        .catch((err) => {
          console.warn('Survival API error:', err);
          setSurvivalData(null);
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

  // Backend is the single source of truth - no browser-side risk formulas
  const activeRisk = backendData?.risk ?? null;
  const activeBand = backendData
    ? { label: backendData.risk_band, tone: backendData.risk_band.toLowerCase() }
    : { label: 'Unavailable', tone: 'neutral' };
  const activePct = activeRisk !== null ? Math.round(activeRisk * 100) : null;

  // Counterfactual reverse goal-seeking computation (using backend model predictions)
  const counterfactualPlan = useMemo(() => {
    if (!backendData) return null;
    return generateCounterfactualPlan(
      patient,
      backendData.risk,
      backendData.controlled_risk,
      targetRiskGoal
    );
  }, [patient, backendData, targetRiskGoal]);

  // Real Cox model survival trajectory returned by backend
  const survivalTrajectory = survivalData;

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

  // Fetch AI-generated care plan from Gemini backend
  const fetchAIGuidance = useCallback(() => {
    setLoadingAI(true);
    setAiError(null);
    fetch('/api/ai-guidance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...patient,
        riskScore: activeRisk,
        riskBand: activeBand.label,
      }),
    })
      .then((res) => res.json())
      .then((data) => {
        setLoadingAI(false);
        if (data.success && data.carePlan) {
          setAiGuidance(data.carePlan);
          showToast('✨ AI-powered care plan generated by Gemini!');
        } else {
          setAiError(data.error || 'AI guidance unavailable.');
          showToast(data.error || 'Could not generate AI plan.', 'error');
        }
      })
      .catch((err) => {
        setLoadingAI(false);
        setAiError('Failed to reach the AI guidance service.');
        showToast('AI service unreachable. Check your API key.', 'error');
      });
  }, [patient, activeRisk, activeBand, showToast]);

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

  // ── COPILOT SEND MESSAGE ───────────────────────────────────────────────
  const handleChatSend = useCallback(async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    const userMsg = { role: 'user', content: msg };
    const history = [...chatMessages, userMsg];
    setChatMessages(history);
    setChatInput('');
    setChatLoading(true);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 60);

    try {
      const res = await fetch('/api/copilot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: msg,
          patient: { ...patient, risk: backendData?.risk ?? 0.2 },
          history: chatMessages.slice(-6),
        }),
      });
      const data = await res.json();
      const reply = data.reply || data.error || 'Sorry, I could not get a response. Please try again.';
      setChatMessages([...history, { role: 'assistant', content: reply }]);
    } catch (err) {
      setChatMessages([...history, { role: 'assistant', content: '⚠️ Connection error. Please ensure the backend is running on port 5000.' }]);
    } finally {
      setChatLoading(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 60);
    }
  }, [chatInput, chatMessages, chatLoading, patient, backendData]);

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
          </div>

          {/* MAIN RISK CARD */}
          <div className={`risk-card ${backendOnline ? activeBand.tone : 'neutral'}`}>
            <div className="risk-header">
              <span className="label">Predicted Study Mortality Risk (~32 Years Follow-up)</span>
              {loading && <span className="calculating-badge">Scoring Patient...</span>}
            </div>

            {backendOnline && activePct !== null ? (
              <>
                <div className="risk-value">{activePct}%</div>

                <div className="risk-band-row">
                  <span className={`risk-band-tag ${activeBand.tone}`}>{activeBand.label} Risk Tier</span>
                  {backendData?.flagged && <span className="flag-alert">⚠️ High Risk Flag</span>}
                </div>

                <div className="risk-bar">
                  <span style={{ width: `${Math.min((activeRisk || 0) * 100, 100)}%` }} />
                </div>

                <p className="risk-urgency-note">
                  {insights?.urgency || (activeBand.label === 'High' ? 'High Priority Clinical Alert' : 'Routine Monitoring')}
                </p>
              </>
            ) : (
              <div style={{ padding: '16px 0' }}>
                <div className="risk-value" style={{ fontSize: '1.4rem', color: '#f87171' }}>
                  Model service unavailable
                </div>
                <p className="risk-urgency-note" style={{ color: '#fca5a5', marginTop: '8px' }}>
                  Backend model server is offline or unreachable. Please verify server is running on port 5000.
                </p>
              </div>
            )}

            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '8px', textAlign: 'center' }}>
              ⚠️ Educational project, not medical advice.
            </div>

            {/* Quick Save button in Hero Card */}
            <div className="hero-quick-save">
              <button
                className="btn-save-record"
                onClick={handleSavePatient}
                disabled={savingRecord || !backendOnline}
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

              {/* LONGITUDINAL SURVIVAL TRAJECTORY CURVE */}
              <div style={{ marginTop: '24px' }}>
                <SurvivalTrajectoryCurve trajectoryData={survivalTrajectory} patient={patient} />
              </div>
            </div>
          </section>
        )}

        {/* TAB 3: PATIENT CARE PLAN & GUIDANCE (WITH INTERACTIVE COUNTERFACTUAL ENGINE) */}
        {activeTab === 'guidance' && (
          <section className="guidance-section">
            <div className="panel guidance-panel">
              <div className="panel-header">
                <div>
                  <h3>Interactive Counterfactual Care Plan & Goal-Seeking Engine</h3>
                  <span>Reverse Risk Optimization & Prescriptive Medicine</span>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <button
                    className="btn-primary-action"
                    onClick={() => { setAiGuidance(null); fetchAIGuidance(); }}
                    disabled={loadingAI}
                    title="Generate a fully personalized AI care plan using Google Gemini"
                    style={{ fontSize: '0.85rem', padding: '8px 16px' }}
                  >
                    <span>✨</span>
                    <span>{loadingAI ? 'Gemini is Thinking...' : 'Generate AI Care Plan'}</span>
                  </button>
                  <button
                    className="btn-secondary-action"
                    onClick={() => window.print()}
                    title="Print official clinical care plan"
                  >
                    🖨️ Print Clinical Plan
                  </button>
                </div>
              </div>

              {/* AI ERROR BANNER */}
              {aiError && (
                <div style={{
                  background: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.4)',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  marginBottom: '16px',
                  color: '#fca5a5',
                  fontSize: '0.85rem'
                }}>
                  ⚠️ <strong>AI Guidance Error:</strong> {aiError}
                  <br />
                  <small style={{ color: '#9bb8dc' }}>Make sure your Gemini API key (starting with AIzaSy...) is correctly set in <code>backend/.env</code> and restart the backend server.</small>
                </div>
              )}

              {/* AI LOADING STATE */}
              {loadingAI && (
                <div style={{
                  background: 'rgba(99, 102, 241, 0.1)',
                  border: '1px solid rgba(99, 102, 241, 0.3)',
                  borderRadius: '10px',
                  padding: '24px',
                  textAlign: 'center',
                  marginBottom: '16px'
                }}>
                  <div style={{ fontSize: '2rem', marginBottom: '8px' }}>✨</div>
                  <strong style={{ color: '#a5b4fc' }}>Google Gemini is generating a personalized care plan...</strong>
                  <p style={{ color: '#8ea8c8', fontSize: '0.82rem', margin: '4px 0 0' }}>Analyzing patient vitals, risk profile, and clinical guidelines. This takes 3–5 seconds.</p>
                </div>
              )}

              {/* AI GENERATED PLAN BANNER */}
              {aiGuidance && !loadingAI && (
                <div style={{
                  background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(16,185,129,0.08))',
                  border: '1px solid rgba(99,102,241,0.35)',
                  borderRadius: '10px',
                  padding: '14px 18px',
                  marginBottom: '16px',
                  display: 'flex',
                  gap: '12px',
                  alignItems: 'flex-start'
                }}>
                  <span style={{ fontSize: '1.6rem' }}>✨</span>
                  <div>
                    <strong style={{ color: '#a5b4fc', fontSize: '0.95rem' }}>AI-Powered Plan — Powered by Google Gemini 1.5 Flash</strong>
                    <p style={{ color: '#9bb8dc', fontSize: '0.82rem', margin: '4px 0 0' }}>
                      {aiGuidance.riskSummary}
                    </p>
                    {aiGuidance.primaryGoal && (
                      <p style={{ color: '#6ee7b7', fontSize: '0.82rem', margin: '6px 0 0' }}>
                        🎯 <strong>Primary Goal:</strong> {aiGuidance.primaryGoal}
                      </p>
                    )}
                  </div>
                </div>
              )}

              {/* COUNTERFACTUAL GOAL SEEKER HERO */}
              <div className="counterfactual-hero">
                <div className="counterfactual-top">
                  <div className="counterfactual-header">
                    <h4>
                      <span>🎯</span>
                      <span>Target Risk Goal-Seeker</span>
                    </h4>
                    <p>
                      Set an achievable target risk for this patient. The engine reverse-calculates the exact
                      minimum biometric reductions needed and prescribes tailored lifestyle, nutrition, and exercise regimens.
                    </p>
                  </div>

                  <div className="goal-controls">
                    <span className="goal-label">Select Target Heart Risk Goal:</span>
                    <div className="goal-chips-row">
                      <button
                        className={`goal-chip ${targetRiskGoal === 20 ? 'active' : ''}`}
                        onClick={() => { setTargetRiskGoal(20); showToast('Target set to <20% (Low Risk Threshold)'); }}
                      >
                        Target &lt;20% (Low Risk)
                      </button>
                      <button
                        className={`goal-chip ${targetRiskGoal === 15 ? 'active' : ''}`}
                        onClick={() => { setTargetRiskGoal(15); showToast('Target set to <15% (Optimal)'); }}
                      >
                        Target &lt;15% (Optimal)
                      </button>
                      <button
                        className={`goal-chip ${targetRiskGoal === 10 ? 'active' : ''}`}
                        onClick={() => { setTargetRiskGoal(10); showToast('Target set to <10% (Minimal Hazard)'); }}
                      >
                        Target &lt;10%
                      </button>
                    </div>
                  </div>
                </div>

                {/* PROJECTION COMPARISON BANNER */}
                <div className="projection-comparison">
                  <div className="proj-stat current">
                    <span>Current Baseline Hazard</span>
                    <strong>{activePct}%</strong>
                    <small style={{ color: '#ff9999' }}>{activeBand.label} Risk Tier</small>
                  </div>

                  <div className="proj-stat target">
                    <span>Simulated Target Hazard</span>
                    <strong>{counterfactualPlan ? counterfactualPlan.simulatedRiskPct : activePct}%</strong>
                    <small style={{ color: '#73f2a7' }}>Achievable with Protocol</small>
                  </div>

                  <div className="proj-stat savings">
                    <span>Potential Hazard Reduction</span>
                    <strong>
                      {counterfactualPlan && counterfactualPlan.riskReductionPct > 0
                        ? `-${counterfactualPlan.riskReductionPct}%`
                        : '0% (At Goal)'}
                    </strong>
                    <small style={{ color: '#90c2ff' }}>Absolute Risk Reduction</small>
                  </div>
                </div>

                {/* BIOMETRIC DELTAS ROADMAP */}
                {counterfactualPlan && counterfactualPlan.deltas.length > 0 && (
                  <div className="deltas-section">
                    <div className="deltas-title">
                      <h4>Required Biomarker Reductions (Clinical Delta Roadmap):</h4>
                    </div>
                    <div className="deltas-grid">
                      {counterfactualPlan.deltas.map((d, idx) => (
                        <div key={idx} className="delta-card">
                          <div className="delta-header">
                            <span className="delta-metric">{d.metric}</span>
                            <span className="delta-pill">{d.diff}</span>
                          </div>
                          <div className="delta-values">
                            <span>Current: {d.current}</span>
                            <span>➔</span>
                            <strong>Target: {d.target}</strong>
                          </div>
                          <p className="delta-impact">{d.impact}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* DOCTOR'S PRESCRIPTIVE RECOMMENDATIONS GRID */}
              <div className="prescriptions-grid">
                {/* 1. DIETARY PRESCRIPTION */}
                <div className="prescription-card diet">
                  <div className="prescript-header">
                    <div>
                      <h4>🥗 Targeted Nutrition Prescription</h4>
                      <span style={{ fontSize: '0.8rem', color: '#93c5fd' }}>
                        {aiGuidance?.dietPlan?.title || counterfactualPlan?.dietPlan?.title || 'Cardiometabolic Nutrition Plan'}
                      </span>
                    </div>
                    <span className="prescript-tag">
                      {aiGuidance?.dietPlan?.sodiumTarget || counterfactualPlan?.dietPlan?.sodium || 'Sodium: < 1,500 mg/day'}
                    </span>
                  </div>

                  <ul className="prescript-list">
                    {(aiGuidance?.dietPlan?.guidelines || counterfactualPlan?.dietPlan?.guidelines || []).map((g, idx) => (
                      <li key={idx}>{g}</li>
                    ))}
                  </ul>

                  {/* AI-only: meal timing and sample meal */}
                  {aiGuidance?.dietPlan?.mealTiming && (
                    <p style={{ margin: '10px 0 4px', fontSize: '0.82rem', color: '#6ee7b7' }}>
                      🕐 <strong>Meal Timing:</strong> {aiGuidance.dietPlan.mealTiming}
                    </p>
                  )}
                  {aiGuidance?.dietPlan?.sampleMeal && (
                    <div style={{
                      background: 'rgba(16,185,129,0.08)',
                      borderRadius: '8px',
                      padding: '10px 12px',
                      marginTop: '8px',
                      fontSize: '0.8rem',
                      color: '#a7f3d0',
                      border: '1px solid rgba(16,185,129,0.2)'
                    }}>
                      🍽️ <strong>Sample Day Meal Plan:</strong><br/>{aiGuidance.dietPlan.sampleMeal}
                    </div>
                  )}
                </div>

                {/* 2. EXERCISE SCHEDULE PRESCRIPTION */}
                <div className="prescription-card exercise">
                  <div className="prescript-header">
                    <div>
                      <h4>🏃‍♂️ Prescriptive Exercise Protocol & Weekly Schedule</h4>
                      <span style={{ fontSize: '0.8rem', color: '#93c5fd' }}>
                        {aiGuidance?.exercisePlan?.weeklyMinutes || counterfactualPlan?.exercisePlan?.duration || '150–180 min / week'}
                      </span>
                    </div>
                    <span className="prescript-tag" style={{ background: 'rgba(59, 130, 246, 0.2)', borderColor: 'rgba(59, 130, 246, 0.5)', color: '#93c5fd' }}>
                      {aiGuidance?.exercisePlan?.intensity || '5 Days / Week'}
                    </span>
                  </div>

                  <div className="exercise-schedule-grid">
                    {(aiGuidance?.exercisePlan?.schedule || counterfactualPlan?.exercisePlan?.schedule || []).map((item, idx) => (
                      <div key={idx} className="exercise-day-item">
                        <span className="exercise-day-badge">{item.day}</span>
                        <span className="exercise-day-desc">{item.activity}</span>
                      </div>
                    ))}
                  </div>
                  {aiGuidance?.exercisePlan?.progressionTip && (
                    <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#93c5fd' }}>
                      📈 <strong>Progression:</strong> {aiGuidance.exercisePlan.progressionTip}
                    </p>
                  )}
                  {!aiGuidance && (
                    <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#8bb0d6' }}>
                      💡 <strong>Clinical Mechanism:</strong> {counterfactualPlan?.exercisePlan?.impact}
                    </p>
                  )}
                </div>

                {/* 3. LIFESTYLE CHANGES (AI-only) */}
                {aiGuidance?.lifestyleChanges && aiGuidance.lifestyleChanges.length > 0 && (
                  <div className="prescription-card" style={{ background: 'rgba(251,191,36,0.06)', borderColor: 'rgba(251,191,36,0.2)' }}>
                    <div className="prescript-header">
                      <div>
                        <h4>🌿 Lifestyle Modifications</h4>
                        <span style={{ fontSize: '0.8rem', color: '#fde68a' }}>Behavioral & Habit Changes</span>
                      </div>
                      <span className="prescript-tag" style={{ background: 'rgba(251,191,36,0.15)', borderColor: 'rgba(251,191,36,0.4)', color: '#fde68a' }}>
                        ✨ AI-Generated
                      </span>
                    </div>
                    <ul className="prescript-list">
                      {aiGuidance.lifestyleChanges.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* 4. CLINICAL & MEDICAL ACTION PROTOCOL */}
                <div className="prescription-card full-width">
                  <div className="prescript-header">
                    <div>
                      <h4>💊 Clinical Interventions & Physician Actions</h4>
                      <span style={{ fontSize: '0.8rem', color: '#c4b5fd' }}>
                        Guideline-Directed Medical Therapy (GDMT)
                      </span>
                    </div>
                    <span className="prescript-tag" style={{ background: 'rgba(139, 92, 246, 0.2)', borderColor: 'rgba(139, 92, 246, 0.5)', color: '#c4b5fd' }}>
                      Physician Review
                    </span>
                  </div>

                  <ul className="prescript-list">
                    {(aiGuidance?.medicalGuidance || counterfactualPlan?.clinicalActions || []).map((act, idx) => (
                      <li key={idx}>{act}</li>
                    ))}
                  </ul>

                  {/* AI-only: Target Biomarkers section */}
                  {aiGuidance?.targetBiomarkers && (
                    <div style={{
                      marginTop: '16px',
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
                      gap: '10px'
                    }}>
                      {Object.entries(aiGuidance.targetBiomarkers).map(([key, value]) => (
                        <div key={key} style={{
                          background: 'rgba(139,92,246,0.1)',
                          border: '1px solid rgba(139,92,246,0.25)',
                          borderRadius: '8px',
                          padding: '10px 12px'
                        }}>
                          <span style={{ fontSize: '0.72rem', color: '#c4b5fd', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            {key.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase())}
                          </span>
                          <strong style={{ display: 'block', color: '#ede9fe', fontSize: '0.88rem', marginTop: '4px' }}>{value}</strong>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* FOOTER ACTION BAR */}
              <div className="care-plan-footer-actions">
                <div>
                  <strong style={{ color: '#ffffff' }}>Ready to hand this plan to the patient?</strong>
                  <p style={{ margin: '2px 0 0', fontSize: '0.8rem', color: '#9bb8dc' }}>
                    You can print this customized prescription plan or save it into the patient's record history.
                  </p>
                </div>
                <div style={{ display: 'flex', gap: '10px' }}>
                  <button className="btn-secondary-action" onClick={() => window.print()}>
                    🖨️ Print Prescription
                  </button>
                  <button className="btn-save-record" onClick={handleSavePatient} disabled={savingRecord}>
                    💾 Save Plan to History
                  </button>
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

      {/* ── CLINICAL AI COPILOT (FIXED RIGHT CORNER ICON) ───────────────── */}

      {/* Floating Copilot Action Button - Fixed Position */}
      <button
        id="copilot-fab"
        className={`copilot-fab ${chatOpen ? 'open' : ''}`}
        onClick={() => {
          const next = !chatOpen;
          setChatOpen(next);
          if (next) {
            setTimeout(() => {
              chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
              chatInputRef.current?.focus();
            }, 100);
          }
        }}
        title={chatOpen ? 'Close AI Copilot' : 'Open Clinical AI Copilot'}
        aria-label="Clinical AI Copilot"
      >
        <span className="copilot-fab-icon">{chatOpen ? '✕' : '🩺'}</span>
        {!chatOpen && <span className="copilot-fab-pulse" />}
      </button>

      {/* Floating Chatbot Popup Drawer (Only visible when toggled) */}
      {chatOpen && (
        <div className="copilot-drawer" role="dialog" aria-label="Clinical AI Copilot Chat">
        {/* Header */}
        <div className="copilot-header">
          <div className="copilot-header-info">
            <span className="copilot-avatar">🩺</span>
            <div>
              <strong>Clinical AI Copilot</strong>
              <span className="copilot-subtitle">Doctor Assistant · Heart Risk Studio</span>
            </div>
          </div>
          <div className="copilot-header-actions">
            <button
              className="copilot-clear-btn"
              onClick={() => setChatMessages([
                { role: 'assistant', content: '💬 Conversation cleared. How can I assist you, Doctor?' }
              ])}
              title="Clear conversation"
            >🗑️</button>
            <button className="copilot-close-btn" onClick={() => setChatOpen(false)} aria-label="Close">✕</button>
          </div>
        </div>

        {/* Patient Context Banner */}
        <div className="copilot-context-bar">
          <span className="ctx-badge">👤 {patient.name || 'Current Patient'}</span>
          <span className="ctx-badge">{patient.Systolic}/{patient.Diastolic} mmHg</span>
          <span className="ctx-badge">{patient.Cholesterol} mg/dL Chol</span>
          <span className={`ctx-badge risk-ctx-${(backendData?.risk_band || 'low').toLowerCase()}`}>
            {backendData ? `${Math.round((backendData.risk || 0.2) * 100)}% Risk` : 'Risk: Pending'}
          </span>
        </div>

        {/* Messages */}
        <div className="copilot-messages" id="copilot-messages-container">
          {chatMessages.map((msg, i) => (
            <div key={i} className={`copilot-msg ${msg.role}`}>
              {msg.role === 'assistant' && <span className="msg-avatar">🩺</span>}
              <div
                className="msg-bubble"
                dangerouslySetInnerHTML={{
                  __html: msg.content
                    // markdown → HTML (minimal inline renderer)
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.+?)\*/g, '<em>$1</em>')
                    .replace(/^### (.+)$/gm, '<h4>$1</h4>')
                    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
                    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
                    .replace(/^---$/gm, '<hr/>')
                    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
                    .replace(/^- (.+)$/gm, '<li>$1</li>')
                    .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
                    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    .replace(/\n\n/g, '</p><p>')
                    .replace(/\n/g, '<br/>')
                    .replace(/^(?!<)(.+)$/gm, '<p>$1</p>')
                }}
              />
            </div>
          ))}

          {chatLoading && (
            <div className="copilot-msg assistant">
              <span className="msg-avatar">🩺</span>
              <div className="msg-bubble typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Quick Prompt Pills */}
        <div className="copilot-quick-prompts">
          {[
            'Explain the pulse pressure flag',
            'Best medications for this patient?',
            'Dietary changes to lower systolic?',
            'Interpret the risk score',
          ].map((q) => (
            <button
              key={q}
              className="quick-prompt-pill"
              onClick={() => {
                setChatInput(q);
                chatInputRef.current?.focus();
              }}
            >{q}</button>
          ))}
        </div>

        {/* Input Area */}
        <div className="copilot-input-area">
          <textarea
            ref={chatInputRef}
            id="copilot-input"
            className="copilot-input"
            placeholder="Ask a clinical question about this patient..."
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSend();
              }
            }}
            rows={2}
            disabled={chatLoading}
          />
          <button
            id="copilot-send-btn"
            className="copilot-send-btn"
            onClick={handleChatSend}
            disabled={chatLoading || !chatInput.trim()}
            aria-label="Send message"
          >
            {chatLoading ? '⏳' : '➤'}
          </button>
        </div>
      </div>
      )}

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
