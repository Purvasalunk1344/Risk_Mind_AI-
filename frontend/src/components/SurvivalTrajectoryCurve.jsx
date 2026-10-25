import React, { useState, useMemo } from 'react';

/**
 * SurvivalTrajectoryCurve: Interactive Cox Proportional Hazards Predicted Survival & Risk Trajectory
 * Compares:
 *   1. "Your trajectory if untreated" (Baseline risk profile)
 *   2. "Scenario if blood pressure & smoking are controlled" (BP <= 120/80 mmHg, 0 cigs/day)
 * Displays longitudinal time-to-event progression at Years 1, 3, 5, 10, 15, and 32.
 *
 * NOTE: Illustrative scenario, not a predicted treatment effect (the model measures association, not causation).
 * Survivors are assumed censored at 32 years; the dataset has no follow-up end date.
 * Educational project, not medical advice.
 */
export default function SurvivalTrajectoryCurve({ trajectoryData, patient }) {
  const [viewMode, setViewMode] = useState('hazard'); // 'hazard' (Predicted cumulative risk %) | 'survival' (Predicted survival %)
  const [selectedYear, setSelectedYear] = useState(10);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const trajectory = trajectoryData?.trajectory || [];
  const maxYear = useMemo(() => {
    if (!trajectory.length) return 32;
    return Math.max(...trajectory.map((p) => p.year), 32);
  }, [trajectory]);

  // Find the selected milestone data point (default year 10)
  const activePoint = useMemo(() => {
    const focusYear = hoveredPoint !== null ? hoveredPoint : selectedYear;
    return (
      trajectory.find((p) => p.year === focusYear) ||
      trajectory.find((p) => p.year === 10) ||
      trajectory[trajectory.length - 1]
    );
  }, [trajectory, selectedYear, hoveredPoint]);

  // SVG dimensions
  const svgWidth = 840;
  const svgHeight = 330;
  const margin = { top: 35, right: 50, bottom: 45, left: 60 };
  const innerWidth = svgWidth - margin.left - margin.right;
  const innerHeight = svgHeight - margin.top - margin.bottom;

  // Scale mappings: X = Years 0 to maxYear (32), Y = 0% to 100%
  const getX = (year) => margin.left + (year / maxYear) * innerWidth;
  const getY = (val) => margin.top + innerHeight - (Math.min(Math.max(val, 0), 100) / 100.0) * innerHeight;

  // Generate SVG path for a given key
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

  // Generate shaded area between the two curves (without forced positive clamping)
  const createAreaPath = () => {
    if (trajectory.length < 2) return '';
    const uKey = viewMode === 'hazard' ? 'untreatedRisk' : 'untreatedSurvival';
    const cKey = viewMode === 'hazard' ? 'controlledRisk' : 'controlledSurvival';

    let d = `M ${getX(trajectory[0].year)} ${getY(trajectory[0][uKey])}`;
    for (let i = 1; i < trajectory.length; i++) {
      d += ` L ${getX(trajectory[i].year)} ${getY(trajectory[i][uKey])}`;
    }
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
  const yearMilestones = [1, 3, 5, 10, 15, 32];

  const currentBp = `${patient?.Systolic || 120}/${patient?.Diastolic || 80} mmHg`;
  const currentSmoking = patient?.Smoking > 0 ? `${patient.Smoking} cigs/day` : 'Non-smoker';

  return (
    <div className="survival-trajectory-container">
      {/* DISCLAIMER BANNER */}
      <div style={{
        padding: '10px 16px',
        background: 'rgba(239, 68, 68, 0.1)',
        border: '1px solid rgba(239, 68, 68, 0.3)',
        borderRadius: '8px',
        marginBottom: '16px',
        fontSize: '12px',
        color: '#fca5a5',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <span style={{ fontSize: '15px' }}>⚠️</span>
        <div>
          <strong>Educational project, not medical advice.</strong> All outputs are statistical associations derived from historical SASHELP.HEART data. Discuss any health concerns with a qualified clinician.
        </div>
      </div>

      {/* SECTION HEADER & VIEW TOGGLE */}
      <div className="st-header">
        <div className="st-header-title">
          <div className="st-badge">
            <span className="st-pulse-dot" />
            <span>Time-to-Event Prognosis (Cox Proportional Hazards Model)</span>
          </div>
          <h4>Longitudinal Survival & Risk Trajectory</h4>
          <p className="st-subtitle">
            Longitudinal survival modeled by Cox PH across follow-up time points (Years 1, 3, 5, 10, 15, and 32).
            Survivors are assumed censored at 32 years; the dataset has no follow-up end date.
          </p>
        </div>

        {/* MODE TOGGLE */}
        <div className="st-toggle-group">
          <button
            type="button"
            className={`st-toggle-btn ${viewMode === 'hazard' ? 'active' : ''}`}
            onClick={() => setViewMode('hazard')}
          >
            📈 Predicted cumulative risk
          </button>
          <button
            type="button"
            className={`st-toggle-btn ${viewMode === 'survival' ? 'active' : ''}`}
            onClick={() => setViewMode('survival')}
          >
            🛡️ Predicted survival (Cox model)
          </button>
        </div>
      </div>

      {/* COMPARATIVE LEGEND */}
      <div className="st-legend-bar">
        <div className="st-legend-item untreated">
          <span className="st-legend-line red" />
          <span className="st-legend-dot red" />
          <div>
            <strong>Untreated baseline</strong>
            <span className="st-legend-sub">
              Baseline parameters ({currentBp}, {currentSmoking})
            </span>
          </div>
        </div>

        <div className="st-legend-item controlled">
          <span className="st-legend-line green" />
          <span className="st-legend-dot green" />
          <div>
            <strong>Controlled scenario (BP ≤ 120/80, 0 cigs)</strong>
            <span className="st-legend-sub">
              Illustrative scenario, not a predicted treatment effect (the model measures association, not causation)
            </span>
          </div>
        </div>

        <div className="st-legend-item benefit">
          <span className="st-legend-swatch" />
          <div>
            <strong>Difference in the scenario</strong>
            <span className="st-legend-sub">Model-predicted gap between profiles</span>
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
            <linearGradient id="stBenefitGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.20" />
              <stop offset="100%" stopColor="#059669" stopOpacity="0.35" />
            </linearGradient>

            <linearGradient id="stUntreatedGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>

            <linearGradient id="stControlledGrad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>

            <filter id="stRedGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3.5" floodColor="#ef4444" floodOpacity="0.5" />
            </filter>
            <filter id="stGreenGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="2" stdDeviation="3.5" floodColor="#10b981" floodOpacity="0.5" />
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

          {/* X-AXIS GRID LINES & MILESTONE LABELS */}
          {[0, 1, 3, 5, 10, 15, 32].map((year) => {
            const x = getX(year);
            const isSelected = (hoveredPoint !== null ? hoveredPoint : selectedYear) === year;
            return (
              <g key={year} className="st-x-grid">
                <line
                  x1={x}
                  y1={margin.top}
                  x2={x}
                  y2={svgHeight - margin.bottom}
                  stroke={isSelected ? 'rgba(56, 189, 248, 0.35)' : 'rgba(255, 255, 255, 0.06)'}
                  strokeDasharray="3 3"
                />
                <text
                  x={x}
                  y={svgHeight - margin.bottom + 22}
                  textAnchor="middle"
                  fill={isSelected ? '#38bdf8' : 'rgba(148, 163, 184, 0.9)'}
                  fontSize="11"
                  fontWeight={isSelected || year === 10 ? '700' : '500'}
                  fontFamily="Inter, sans-serif"
                >
                  {year === 0 ? 'Baseline' : `Yr ${year}`}
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
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#stRedGlow)"
          />

          {/* CONTROLLED LINE */}
          <path
            d={controlledPath}
            fill="none"
            stroke="url(#stControlledGrad)"
            strokeWidth="3"
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
                  r={isFocused ? 6.5 : 4.5}
                  fill="#ef4444"
                  stroke="#ffffff"
                  strokeWidth={isFocused ? 2 : 1}
                />

                {/* Controlled Dot */}
                <circle
                  cx={x}
                  cy={yControlled}
                  r={isFocused ? 6.5 : 4.5}
                  fill="#10b981"
                  stroke="#ffffff"
                  strokeWidth={isFocused ? 2 : 1}
                />

                {/* Value pill if focused */}
                {isFocused && (
                  <>
                    <rect
                      x={x - 24}
                      y={yUntreated - 22}
                      width="48"
                      height="18"
                      rx="4"
                      fill="#1e1b2e"
                      stroke="#ef4444"
                      strokeWidth="1"
                    />
                    <text
                      x={x}
                      y={yUntreated - 9}
                      textAnchor="middle"
                      fill="#fca5a5"
                      fontSize="10"
                      fontWeight="700"
                    >
                      {point[untreatedKey]}%
                    </text>

                    <rect
                      x={x - 24}
                      y={yControlled + 6}
                      width="48"
                      height="18"
                      rx="4"
                      fill="#06251b"
                      stroke="#10b981"
                      strokeWidth="1"
                    />
                    <text
                      x={x}
                      y={yControlled + 19}
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

      {/* MILESTONE SELECTOR STRIP */}
      <div className="st-milestones-row">
        {yearMilestones.map((yr) => {
          const pt = trajectory.find((p) => p.year === yr);
          if (!pt) return null;
          const isSelected = activePoint?.year === yr;
          const diff = pt.difference !== undefined ? pt.difference : Math.round((pt.controlledSurvival - pt.untreatedSurvival) * 10) / 10;

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
                  {diff >= 0 ? `+${diff}%` : `${diff}%`} Diff
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

      {/* SELECTED MILESTONE DETAILS CARD */}
      {activePoint && (
        <div className="st-inspector-box">
          <div className="st-inspector-header">
            <div className="st-inspector-title">
              <span className="st-inspector-icon">🎯</span>
              <div>
                <h5>
                  Year {activePoint.year} Horizon
                </h5>
                <span className="st-inspector-sub">
                  Predicted by Cox Proportional Hazards Model (assumed right-censored at 32 years)
                </span>
              </div>
            </div>

            <div className="st-inspector-gain-pill">
              <span className="st-gain-label">Scenario Difference</span>
              <span className="st-gain-val">
                {activePoint.difference !== undefined
                  ? `${activePoint.difference >= 0 ? '+' : ''}${activePoint.difference}%`
                  : `${Math.round((activePoint.controlledSurvival - activePoint.untreatedSurvival) * 10) / 10}%`}
              </span>
            </div>
          </div>

          <div className="st-inspector-grid">
            <div className="st-inspector-col untreated">
              <div className="st-col-title">
                <span className="st-status-circle red" />
                <span>Untreated Profile</span>
              </div>
              <div className="st-col-stat">
                <span className="number">{activePoint.untreatedRisk}%</span>
                <span className="label">Predicted cumulative risk</span>
              </div>
              <div className="st-col-substat">
                Predicted Survival: <strong>{activePoint.untreatedSurvival}%</strong>
              </div>
            </div>

            <div className="st-inspector-col controlled">
              <div className="st-col-title">
                <span className="st-status-circle green" />
                <span>Controlled Scenario (BP ≤ 120/80, 0 cigs)</span>
              </div>
              <div className="st-col-stat">
                <span className="number">{activePoint.controlledRisk}%</span>
                <span className="label">Predicted cumulative risk</span>
              </div>
              <div className="st-col-substat">
                Predicted Survival: <strong>{activePoint.controlledSurvival}%</strong>
              </div>
            </div>

            <div className="st-inspector-col summary">
              <div className="st-col-title">
                <span className="st-status-circle blue" />
                <span>Scenario Interpretation</span>
              </div>
              <p className="st-interpret-text">
                In this illustrative scenario where blood pressure is maintained at ≤120/80 mmHg with zero tobacco exposure, the model-predicted cumulative event risk at Year {activePoint.year} is <strong>{activePoint.controlledRisk}%</strong> compared to <strong>{activePoint.untreatedRisk}%</strong> in the untreated baseline profile.
              </p>
              <div className="st-interpret-badge">
                ⚠️ Illustrative scenario, not a predicted treatment effect (the model measures association, not causation). Discuss with a clinician.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
