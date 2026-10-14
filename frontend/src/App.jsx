import { useMemo, useState } from 'react';
import { calculateRisk, getRiskBand, samplePatient } from './data/scoreLogic';

const fieldConfig = [
  { label: 'Age', key: 'AgeAtStart', type: 'number', min: 20, max: 100 },
  { label: 'Sex', key: 'Sex', type: 'select', options: ['Female', 'Male'] },
  { label: 'Systolic', key: 'Systolic', type: 'number', min: 80, max: 220 },
  { label: 'Diastolic', key: 'Diastolic', type: 'number', min: 40, max: 140 },
  { label: 'Cholesterol', key: 'Cholesterol', type: 'number', min: 100, max: 400 },
  { label: 'Weight (lb)', key: 'Weight', type: 'number', min: 80, max: 350 },
  { label: 'Height (in)', key: 'Height', type: 'number', min: 48, max: 84 },
  { label: 'Smoking (packs/day)', key: 'Smoking', type: 'number', min: 0, max: 80 },
];

const formatPct = (value) => `${Math.round(value * 100)}%`;

function App() {
  const [patient, setPatient] = useState(samplePatient);

  const risk = useMemo(() => calculateRisk(patient), [patient]);
  const band = useMemo(() => getRiskBand(risk), [risk]);

  const updateField = (key, value) => {
    setPatient((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const metrics = [
    { label: 'BMI', value: patient.Weight && patient.Height ? ((patient.Weight * 703) / (patient.Height * patient.Height)).toFixed(1) : '—' },
    { label: 'Pulse Pressure', value: patient.Systolic && patient.Diastolic ? (Number(patient.Systolic) - Number(patient.Diastolic)).toString() : '—' },
    { label: 'Smoking', value: patient.Smoking ? `${patient.Smoking} packs/day` : 'None' },
    { label: 'Band', value: band.label },
  ];

  return (
    <div className="app-shell">
      <div className="ambient ambient-1" />
      <div className="ambient ambient-2" />

      <header className="topbar">
        <div className="brand-wrap">
          <div className="logo-dot" />
          <div>
            <p className="eyebrow">Clinical intelligence</p>
            <h1>HeartRisk Studio</h1>
          </div>
        </div>
        <button className="ghost-button" onClick={() => setPatient(samplePatient)}>
          Load sample patient
        </button>
      </header>

      <main className="dashboard">
        <section className="hero-panel">
          <div className="hero-copy">
            <p className="badge">Champion model preview</p>
            <h2>Personalized cardiovascular risk monitoring</h2>
            <p>
              Explore patient-level risk using the same clinical dimensions used in the
              project’s heart-disease mortality model.
            </p>
          </div>

          <div className={`risk-card ${band.tone}`}>
            <span className="label">Estimated risk</span>
            <div className="risk-value">{formatPct(risk)}</div>
            <div className="risk-band">{band.label}</div>
            <div className="risk-bar">
              <span style={{ width: `${Math.min(risk * 100, 100)}%` }} />
            </div>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel form-panel">
            <div className="panel-header">
              <h3>Patient profile</h3>
              <span>Risk inputs</span>
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
                      value={patient[field.key]}
                      onChange={(e) => updateField(field.key, e.target.value)}
                    />
                  )}
                </label>
              ))}
            </div>
          </div>

          <div className="panel insight-panel">
            <div className="panel-header">
              <h3>Clinical summary</h3>
              <span>Live metrics</span>
            </div>

            <div className="stat-grid">
              {metrics.map((metric) => (
                <div className="stat-box" key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>

            <div className="recommendation-box">
              <p className="mini-label">Recommendation</p>
              <strong>
                {band.label === 'High'
                  ? 'Escalate clinical review and consider preventive intervention.'
                  : band.label === 'Medium'
                    ? 'Monitor closely and reinforce lifestyle and BP optimization.'
                    : 'Current profile is in the lower-risk band; continue routine screening.'}
              </strong>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
