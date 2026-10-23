import React, { useState, useMemo } from 'react';

/**
 * SurvivalTrajectoryCurve: Interactive Kaplan-Meier & Cumulative Event Risk Trajectory
 * Compares:
 *   1. "Your trajectory if untreated" (Baseline risk profile)
 *   2. "Your trajectory if blood pressure & smoking are controlled" (Target BP <= 120/80 mmHg, 0 cigs/day)
 * Displays longitudinal time-to-event progression at Year 1, 3, 5, 10, and 15.
 */
export default function SurvivalTrajectoryCurve({ trajectoryData, patient }) {
  const [viewMode, setViewMode] = useState('hazard'); // 'hazard' (Risk %) | 'survival' (Survival %)
  const [selectedYear, setSelectedYear] = useState(10);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const trajectory = trajectoryData?.trajectory || [];

  // Find the selected milestone data point (default year 10)
  const activePoint = useMemo(() => {
    const focusYear = hoveredPoint !== null ? hoveredPoint : selectedYear;
    return trajectory.find((p) => p.year === focusYear) || trajectory.find((p) => p.year === 10) || trajectory[trajectory.length - 1];
  }, [trajectory, selectedYear, hoveredPoint]);

  // SVG dimensions
  const svgWidth = 820;
  const svgHeight = 330;
  const margin = { top: 35, right: 45, bottom: 45, left: 60 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  // Scale mappings: X = Years 0 to 15, Y = 0% to 100%
  const getX = (year) => margin.left + (year / 15.0) * innerWidth;
  const getY = (val) => margin.top + innerHeight - (Math.min(Math.max(val, 0), 100) / 100.0) * innerHeight;

  // Generate SVG path for a given key ('untreated_risk' / 'untreated_survival' / etc.)
  const createPath = (valueKey) => {
    if (!trajectory.length) return '';
    return trajectory
      .map((p, idx) => {
        const x = getX(p.year);
        const y = getY(p[valueKey]);
        return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
      })
      .join(' ');
  };

  // Generate shaded area between the two curves
  const createAreaPath = () => {
    if (trajectory.length < 2) return '';
    const uKey = viewMode === 'hazard' ? 'untreatedRisk' : 'controlledSurvival';
    const cKey = viewMode === 'hazard' ? 'controlledRisk' : 'untreatedSurvival';

    // Forward path along upper curve
    let d = `M ${getX(trajectory[0].year)} ${getY(trajectory[0][uKey])}`;
    for (let i = 1; i < trajectory.length; i++) {
      d += ` L ${getX(trajectory[i].year)} ${getY(trajectory[i][uKey])}`;
    }
    // Backward path along lower curve
    for (let i = trajectory.length - 1; i >= 0; i--) {
      d += ` L ${getX(trajectory[i].year)} ${getY(trajectory[i][cKey])}`;
    }
    d += ' Z';
    return d;
  };

  const untreatedKey = viewMode === 'hazard' ? 'untreatedRisk' : 'untreatedSurvival';
  const controlledKey = viewMode === 'hazard' ? 'controlledRisk' : 'controlledSurvival';

  const untreatedPath = createPath(untreatedKey);
  const controlledPath = createPath(controlledKey);
  const areaPath = createAreaPath();

  // Y-axis grid marks (0%, 20%, 40%, 60%, 80%, 100%)
  const yTicks = [0, 20, 40, 60, 80, 100];
  const yearMilestones = [1, 3, 5, 10, 15];

  const currentBp = `${patient?.Systolic || 120}/${patient?.Diastolic || 80} mmHg`;
  const currentSmoking = patient?.Smoking > 0 ? `${patient.Smoking} cigs/day` : 'Non-smoker';

  return (
    <div className="survival-trajectory-container">
      {/* SECTION HEADER & VIEW TOGGLE */}
      <div className="st-header">
        <div className="st-header-title">
          <div className="st-badge">
            <span className="st-pulse-dot" />
            <span>Time-to-Event Prognosis (Kaplan-Meier Methodology)</span>
          </div>
          <h4>Longitudinal Risk & Event-Free Survival Trajectory</h4>
          <p className="st-subtitle">
            Evaluates cardiovascular event hazard at <strong>Year 1, 3, 5, 10, and 15</strong> comparing baseline untreated trajectory against complete blood pressure control (≤120/80 mmHg) and smoking cessation.
          </p>
        </div>

        {/* MODE TOGGLE */}
        <div className="st-toggle-group">
          <button
            type="button"
            className={`st-toggle-btn ${viewMode === 'hazard' ? 'active' : ''}`}
            onClick={() => setViewMode('hazard')}
          >
            📈 Cumulative Event Hazard (%)
          </button>
          <button
            type="button"
            className={`st-toggle-btn ${viewMode === 'survival' ? 'active' : ''}`}
            onClick={() => setViewMode('survival')}
          >
            🛡️ Kaplan-Meier Survival Rate (%)
          </button>
        </div>
      </div>

      {/* COMPARATIVE LEGEND & CURRENT VITALS CALLOUT */}
      <div className="st-legend-bar">
        <div className="st-legend-item untreated">
          <span className="st-legend-line red" />
          <span className="st-legend-dot red" />
          <div>
            <strong>Your Trajectory If Untreated</strong>
            <span className="st-legend-sub">
              Baseline parameters ({currentBp}, {currentSmoking})
            </span>
          </div>
        </div>

        <div className="st-legend-item controlled">
          <span className="st-legend-line green" />
          <span className="st-legend-dot green" />
          <div>
            <strong>Your Trajectory If BP & Smoking Are Controlled</strong>
            <span className="st-legend-sub">
              Target BP ≤ 120/80 mmHg + Complete Smoking Cessation (0 cigs)
            </span>
          </div>
        </div>

        <div className="st-legend-item benefit">
          <span className="st-legend-swatch" />
          <div>
            <strong>Survival Advantage Zone</strong>
            <span className="st-legend-sub">Absolute risk averted through intervention</span>
          </div>
        </div>
      </div>

      {/* INTERACTIVE SVG CHART */}
      <div className="st-chart-wrapper">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="st-svg-chart"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Shaded benefit area gradient */}
            <linearGradient id="stBenefitGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#059669" stopOpacity="0.45" />
            </linearGradient>

            {/* Untreated line gradient (Rose Neon) */}
            <linearGradient id="stUntreatedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>

            {/* Controlled line gradient (Emerald Glow) */}
            <linearGradient id="stControlledGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>

            {/* Glow filters */}
            <filter id="stRedGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3.5" floodColor="#ef4444" floodOpacity="0.6" />
            </filter>
            <filter id="stGreenGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3.5" floodColor="#10b981" floodOpacity="0.6" />
            </filter>
          </defs>

          {/* Y-AXIS GRID LINES & LABELS */}
          {yTicks.map((tick) => {
            const y = getY(tick);
            return (
              <g key={tick} className="st-grid-group">
                <line
                  x1={margin.left}
                  y1={y}
                  x2={svgWidth - margin.right}
                  y2={y}
                  stroke="rgba(255, 255, 255, 0.08)"
                  strokeDasharray={tick === 0 || tick === 100 ? 'none' : '3 4'}
                />
                <text
                  x={margin.left - 10}
                  y={y + 4}
                  textAnchor="end"
                  className="st-axis-text"
                  fill="rgba(148, 163, 184, 0.85)"
                  fontSize="11"
                  fontFamily="Inter, sans-serif"
                >
                  {tick}%
                </text>
              </g>
            );
          })}

          {/* CLINICAL THRESHOLD BENCHMARK LINES */}
          {viewMode === 'hazard' && (
            <>
              {/* 20% Clinical threshold */}
              <line
                x1={margin.left}
                y1={getY(20)}
                x2={svgWidth - margin.right}
                y2={getY(20)}
                stroke="#f59e0b"
                strokeWidth="1.2"
                strokeDasharray="4 4"
                strokeOpacity="0.55"
              />
              <text
                x={svgWidth - margin.right - 6}
                y={getY(20) - 5}
                textAnchor="end"
                fill="#f59e0b"
                fontSize="10"
                fontWeight="600"
              >
                20% Moderate Threshold
              </text>

              {/* 50% High risk tier */}
              <line
                x1={margin.left}
                y1={getY(50)}
                x2={svgWidth - margin.right}
                y2={getY(50)}
                stroke="#ef4444"
                strokeWidth="1.2"
                strokeDasharray="4 4"
                strokeOpacity="0.5"
              />
              <text
                x={svgWidth - margin.right - 6}
                y={getY(50) - 5}
                textAnchor="end"
                fill="#ef4444"
                fontSize="10"
                fontWeight="600"
              >
                50% High Hazard Tier
              </text>
            </>
          )}

          {/* X-AXIS GRID LINES & MILESTONE LABELS */}
          {[0, 1, 3, 5, 10, 15].map((year) => {
            const x = getX(year);
            const isSelected = (hoveredPoint !== null ? hoveredPoint : selectedYear) === year;
            return (
              <g key={year} className="st-x-grid">
                <line
                  x1={x}
                  y1={margin.top}
                  x2={x}
                  y2={svgHeight - margin.bottom}
                  stroke={isSelected ? 'rgba(56, 189, 248, 0.3)' : 'rgba(255, 255, 255, 0.05)'}
                  strokeDasharray="3 3"
                />
                <text
                  x={x}
                  y={svgHeight - margin.bottom + 22}
                  textAnchor="middle"
                  fill={isSelected ? '#38bdf8' : 'rgba(148, 163, 184, 0.9)'}
                  fontSize="12"
                  fontWeight={isSelected || year === 10 ? '700' : '500'}
                  fontFamily="Inter, sans-serif"
                >
                  {year === 0 ? 'Baseline' : `Year ${year}`}
                </text>
              </g>
            );
          })}

          {/* SHADED BENEFIT AREA */}
          <path d={areaPath} fill="url(#stBenefitGrad)" />

          {/* UNTREATED LINE */}
          <path
            d={untreatedPath}
            fill="none"
            stroke="url(#stUntreatedGrad)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#stRedGlow)"
          />

          {/* CONTROLLED LINE */}
          <path
            d={controlledPath}
            fill="none"
            stroke="url(#stControlledGrad)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#stGreenGlow)"
          />

          {/* INTERACTIVE MARKERS AT EACH MILESTONE */}
          {trajectory.map((point) => {
            const x = getX(point.year);
            const yUntreated = getY(point[untreatedKey]);
            const yControlled = getY(point[controlledKey]);
            const isFocused = (hoveredPoint !== null ? hoveredPoint : selectedYear) === point.year;

            return (
              <g
                key={point.year}
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedYear(point.year)}
                onMouseEnter={() => setHoveredPoint(point.year)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                {/* Vertical connector line if focused */}
                {isFocused && (
                  <line
                    x1={x}
                    y1={Math.min(yUntreated, yControlled)}
                    x2={x}
                    y2={Math.max(yUntreated, yControlled)}
                    stroke="#38bdf8"
                    strokeWidth="2"
                    strokeDasharray="2 2"
                    opacity="0.8"
                  />
                )}

                {/* Untreated Dot */}
                <circle
                  cx={x}
                  cy={yUntreated}
                  r={isFocused ? 7 : 5}
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth={isFocused ? 2.5 : 1.5}
                />

                {/* Controlled Dot */}
                <circle
                  cx={x}
                  cy={yControlled}
                  r={isFocused ? 7 : 5}
                  fill="#10b981"
                  stroke="#ffffff"
                  strokeWidth={isFocused ? 2.5 : 1.5}
                />

                {/* Value pill on top if focused or key milestones (5, 10, 15) */}
                {(isFocused || point.year === 10) && (
                  <>
                    {/* Untreated value tag */}
                    <rect
                      x={x - 22}
                      y={yUntreated - (viewMode === 'hazard' ? 24 : -8)}
                      width="44"
                      height="18"
                      rx="4"
                      fill="#1e1b2e"
                      stroke="#ef4444"
                      strokeWidth="1"
                    />
                    <text
                      x={x}
                      y={yUntreated - (viewMode === 'hazard' ? 11 : -21)}
                      textAnchor="middle"
                      fill="#fca5a5"
                      fontSize="10"
                      fontWeight="700"
                    >
                      {point[untreatedKey]}%
                    </text>

                    {/* Controlled value tag */}
                    <rect
                      x={x - 22}
                      y={yControlled - (viewMode === 'hazard' ? -8 : 24)}
                      width="44"
                      height="18"
                      rx="4"
                      fill="#06251b"
                      stroke="#10b981"
                      strokeWidth="1"
                    />
                    <text
                      x={x}
                      y={yControlled - (viewMode === 'hazard' ? -21 : 11)}
                      textAnchor="middle"
                      fill="#6ee7b7"
                      fontSize="10"
                      fontWeight="700"
                    >
                      {point[controlledKey]}%
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* INTERACTIVE MILESTONE SELECTOR STRIP */}
      <div className="st-milestones-row">
        {yearMilestones.map((yr) => {
          const pt = trajectory.find((p) => p.year === yr);
          if (!pt) return null;
          const isSelected = activePoint?.year === yr;

          return (
            <button
              key={yr}
              type="button"
              className={`st-milestone-card ${isSelected ? 'active' : ''}`}
              onClick={() => setSelectedYear(yr)}
            >
              <div className="st-card-top">
                <span className="st-year-tag">Year {yr}</span>
                <span className="st-arr-badge">
                  {viewMode === 'hazard' ? `-${pt.arr}% Risk` : `+${pt.survivalGain}% Surv`}
                </span>
              </div>

              <div className="st-card-metrics">
                <div className="st-card-stat red">
                  <span className="st-card-stat-lbl">Untreated</span>
                  <span className="st-card-stat-val">
                    {viewMode === 'hazard' ? `${pt.untreatedRisk}%` : `${pt.untreatedSurvival}%`}
                  </span>
                </div>
                <div className="st-card-stat green">
                  <span className="st-card-stat-lbl">Controlled</span>
                  <span className="st-card-stat-val">
                    {viewMode === 'hazard' ? `${pt.controlledRisk}%` : `${pt.controlledSurvival}%`}
                  </span>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* SELECTED MILESTONE DEEP-DIVE CARD */}
      {activePoint && (
        <div className="st-inspector-box">
          <div className="st-inspector-header">
            <div className="st-inspector-title">
              <span className="st-inspector-icon">🎯</span>
              <div>
                <h5>
                  Year {activePoint.year} Prognostic Milestone:{' '}
                  <span className="st-inspector-highlight">
                    {activePoint.year === 10 ? 'Standard 10-Year Clinical Horizon' : `Year ${activePoint.year} Horizon`}
                  </span>
                </h5>
                <span className="st-inspector-sub">{activePoint.milestone}</span>
              </div>
            </div>

            <div className="st-inspector-gain-pill">
              <span className="st-gain-label">Absolute Risk Averted</span>
              <span className="st-gain-val">-{activePoint.arr}%</span>
            </div>
          </div>

          <div className="st-inspector-grid">
            <div className="st-inspector-col untreated">
              <div className="st-col-title">
                <span className="st-status-circle red" />
                <span>Untreated Trajectory</span>
              </div>
              <div className="st-col-stat">
                <span className="number">{activePoint.untreatedRisk}%</span>
                <span className="label">Cumulative Event Hazard</span>
              </div>
              <div className="st-col-substat">
                Event-Free Survival Probability: <strong>{activePoint.untreatedSurvival}%</strong>
              </div>
            </div>

            <div className="st-inspector-col controlled">
              <div className="st-col-title">
                <span className="st-status-circle green" />
                <span>Controlled Trajectory (BP & Smoking Controlled)</span>
              </div>
              <div className="st-col-stat">
                <span className="number">{activePoint.controlledRisk}%</span>
                <span className="label">Cumulative Event Hazard</span>
              </div>
              <div className="st-col-substat">
                Event-Free Survival Probability: <strong>{activePoint.controlledSurvival}%</strong>
              </div>
            </div>

            <div className="st-inspector-col summary">
              <div className="st-col-title">
                <span className="st-status-circle blue" />
                <span>Clinical Interpretation</span>
              </div>
              <p className="st-interpret-text">
                By maintaining blood pressure at <strong>≤120/80 mmHg</strong> and ensuring <strong>complete tobacco cessation</strong>, the patient cuts their cumulative {activePoint.year}-year risk from <strong>{activePoint.untreatedRisk}%</strong> down to <strong>{activePoint.controlledRisk}%</strong>.
              </p>
              <div className="st-interpret-badge">
                🛡️ Preserves <strong>+{activePoint.survivalGain}%</strong> absolute event-free survival rate over this horizon.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
