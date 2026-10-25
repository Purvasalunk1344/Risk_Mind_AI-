/**
 * scoreLogic.js
 * Frontend clinical display helpers, goal planning, and formatting.
 * NOTE: All risk and survival models run exclusively on the backend (models/*.pkl and Cox PH).
 * The frontend NEVER computes risk or survival formulas itself.
 */

export const samplePatient = {
  AgeAtStart: 58,
  Sex: 'Female',
  Systolic: 145,
  Diastolic: 88,
  Cholesterol: 210,
  Weight: 160,
  Height: 66,
  Smoking: 12,
};

export const getRiskBand = (risk) => {
  if (risk === null || risk === undefined) {
    return { label: 'Unknown', tone: 'neutral' };
  }
  if (risk < 0.20) {
    return { label: 'Low', tone: 'low' };
  } else if (risk < 0.50) {
    return { label: 'Medium', tone: 'medium' };
  } else {
    return { label: 'High', tone: 'high' };
  }
};

/**
 * Generate counterfactual guidance based on backend-scored risk and controlled scenario.
 * Evaluates target scenario modifiable metrics against clinical guidelines.
 */
export const generateCounterfactualPlan = (
  patient,
  currentRisk,
  controlledRisk,
  targetRiskPct = 20
) => {
  if (!patient || currentRisk === null || currentRisk === undefined) return null;

  const currentRiskPct = Math.round(Number(currentRisk) * 100);
  const simRiskPct = controlledRisk !== null && controlledRisk !== undefined
    ? Math.round(Number(controlledRisk) * 100)
    : currentRiskPct;
  const targetPct = Math.max(5, Math.min(targetRiskPct, currentRiskPct));

  const weight = Number(patient.Weight) || 160;
  const height = Number(patient.Height) || 66;
  const systolic = Number(patient.Systolic) || 120;
  const diastolic = Number(patient.Diastolic) || 80;
  const cholesterol = Number(patient.Cholesterol) || 200;
  const smoking = Number(patient.Smoking) || 0;
  const currentBmi = height > 0 ? (weight * 703) / (height * height) : 25;

  // If already at or below target
  if (currentRiskPct <= targetPct) {
    return {
      isAlreadyTarget: true,
      currentRiskPct,
      targetRiskPct: targetPct,
      simulatedRiskPct: currentRiskPct,
      riskReductionPct: 0,
      deltas: [],
      dietPlan: {
        title: "Cardiovascular Nutrition Guidance (DASH / Mediterranean Style)",
        sodium: "Moderate intake; discuss target with a clinician",
        guidelines: [
          "Continue high-fiber whole grain intake (oats, quinoa, brown rice).",
          "Consume 5+ servings of colorful fruits and vegetables daily.",
          "Prioritize healthy unsaturated fats (olive oil, nuts, seeds).",
          "Limit added sugars and ultra-processed foods.",
          "Discuss personalized nutritional needs with a clinician."
        ]
      },
      exercisePlan: {
        title: "Physical Activity Guidance",
        duration: "150 minutes of moderate aerobic activity weekly",
        schedule: [
          { day: "Mon / Wed / Fri", activity: "30 min brisk walk, swimming, or cycling" },
          { day: "Tue / Thu", activity: "20 min light resistance training or bodyweight exercises" },
          { day: "Weekend", activity: "Active recreation: hiking or outdoor stroll" }
        ],
        strength: "2 sessions/week of moderate resistance training",
        impact: "Regular activity supports healthy blood pressure and lipid balance. Discuss with a clinician."
      },
      clinicalActions: [
        "Continue routine wellness checkups and preventive screenings.",
        "Maintain periodic blood pressure logging; discuss readings with a clinician."
      ]
    };
  }

  // Modifiable clinical targets for the controlled scenario
  const simSmoking = 0;
  const simSystolic = Math.min(systolic, 120);
  const simDiastolic = Math.min(diastolic, 80);
  const simCholesterol = Math.min(cholesterol, 190);
  const simWeight = currentBmi > 25 ? Math.round(weight * 0.95) : weight;

  const simulatedPatient = {
    ...patient,
    Systolic: simSystolic,
    Diastolic: simDiastolic,
    Cholesterol: simCholesterol,
    Weight: simWeight,
    Smoking: simSmoking,
  };

  const riskReductionPct = Math.max(0, currentRiskPct - simRiskPct);

  // Compute specific biomarker deltas
  const deltas = [];
  if (smoking > 0) {
    deltas.push({
      metric: 'Smoking Cessation',
      current: `${smoking} cigs/day`,
      target: '0 cigs/day',
      diff: `-${smoking} cigs/day`,
      priority: 'Highest',
      impact: 'Eliminating tobacco exposure lowers cardiovascular risk; discuss cessation strategies with a clinician.'
    });
  }

  if (systolic > 120 || diastolic > 80) {
    deltas.push({
      metric: 'Blood Pressure Control',
      current: `${systolic}/${diastolic} mmHg`,
      target: `${simSystolic}/${simDiastolic} mmHg`,
      diff: `-${systolic - simSystolic}/-${diastolic - simDiastolic} mmHg`,
      priority: 'High',
      impact: 'Lowering blood pressure reduces arterial workload; discuss target levels with a clinician.'
    });
  }

  if (cholesterol > 200) {
    deltas.push({
      metric: 'Serum Cholesterol',
      current: `${cholesterol} mg/dL`,
      target: `${simCholesterol} mg/dL`,
      diff: `-${cholesterol - simCholesterol} mg/dL`,
      priority: 'Medium',
      impact: 'Managing cholesterol helps support arterial health; discuss lipid management with a clinician.'
    });
  }

  if (weight > simWeight) {
    deltas.push({
      metric: 'Body Weight / BMI',
      current: `${weight} lbs (${currentBmi.toFixed(1)} BMI)`,
      target: `${simWeight} lbs (${((simWeight * 703) / (height * height)).toFixed(1)} BMI)`,
      diff: `-${weight - simWeight} lbs`,
      priority: 'Medium',
      impact: 'Gradual weight reduction supports metabolic markers; discuss a tailored plan with a clinician.'
    });
  }

  // Dietary Guidance
  const dietPlan = {
    title: "Cardiovascular Nutrition Guidance (DASH / Mediterranean Style)",
    sodium: systolic >= 140 ? "Sodium reduction recommended; discuss target with a clinician" : "Moderate sodium intake",
    guidelines: [
      "Sodium moderation: Limit processed foods, canned soups, and prepared sauces. Season with herbs and citrus.",
      "Potassium-rich foods: Include vegetables, leafy greens, and beans to support healthy blood pressure.",
      "Heart-healthy fats: Use olive oil in place of saturated fats; include fish or plant-based omega-3 sources.",
      "Fiber intake: Aim for ample dietary fiber from oats, lentils, vegetables, and whole grains.",
      "Limit refined carbohydrates and alcohol intake. Discuss personalized nutritional goals with a clinician."
    ]
  };

  // Exercise Schedule
  const exercisePlan = {
    title: "Physical Activity Guidance",
    duration: "150 minutes of moderate aerobic activity weekly",
    frequency: "Spread across 4–5 days per week, as tolerated",
    schedule: [
      { day: "Monday", activity: "30 min brisk walk (moderate tempo) or low-impact cardio" },
      { day: "Tuesday", activity: "20 min light resistance training or mobility work" },
      { day: "Wednesday", activity: "30 min steady aerobic session (cycling, walking, or swimming)" },
      { day: "Thursday", activity: "Active recovery: gentle stretching and light stroll" },
      { day: "Friday", activity: "30 min moderate aerobic activity" },
      { day: "Saturday", activity: "Light bodyweight or resistance exercises" },
      { day: "Sunday", activity: "Rest and recovery" }
    ],
    strength: "2 weekly sessions of moderate resistance exercise",
    impact: "Regular activity supports healthy arterial function and fitness. Discuss an activity plan with a clinician."
  };

  // Clinical Recommendations
  const clinicalActions = [
    systolic >= 140 || diastolic >= 90
      ? "Blood pressure evaluation: Schedule a clinical review to discuss blood pressure management and target goals."
      : "Blood pressure monitoring: Log resting blood pressure periodically; discuss trends with a clinician.",
    cholesterol >= 240
      ? "Lipid panel review: Schedule a comprehensive fasting lipid panel; discuss management options with a clinician."
      : "Lipid monitoring: Follow up with periodic lipid screening as advised by a healthcare provider.",
    smoking > 0
      ? "Smoking cessation support: Discuss evidence-based cessation strategies and counseling with a clinician."
      : "Tobacco prevention: Maintaining zero tobacco exposure protects vascular health.",
    "Sleep and wellness: Aim for 7–8 hours of restful sleep; discuss any persistent fatigue with a clinician.",
    "Stress management: Incorporate daily relaxation techniques or diaphragmatic breathing to support autonomic balance."
  ];

  return {
    isAlreadyTarget: false,
    currentRiskPct,
    targetRiskPct: targetPct,
    simulatedRiskPct: simRiskPct,
    riskReductionPct,
    simulatedPatient,
    deltas,
    dietPlan,
    exercisePlan,
    clinicalActions,
    disclaimer: "Educational project, not medical advice. Always discuss with a clinician."
  };
};
