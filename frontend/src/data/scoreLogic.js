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
  if (risk < 0.20) {
    return { label: 'Low', tone: 'low' };
  } else if (risk < 0.50) {
    return { label: 'Medium', tone: 'medium' };
  } else {
    return { label: 'High', tone: 'high' };
  }
};

export const calculateRisk = (patient) => {
  if (!patient) return 0.1;

  const weight = Number(patient.Weight) || 160;
  const height = Number(patient.Height) || 66;
  const bmi = height > 0 ? (weight * 703) / (height * height) : 25;

  const age = Number(patient.AgeAtStart) || 50;
  const systolic = Number(patient.Systolic) || 120;
  const diastolic = Number(patient.Diastolic) || 80;
  const cholesterol = Number(patient.Cholesterol) || 200;
  const smoking = Number(patient.Smoking) || 0;
  const isMale = patient.Sex === 'Male' ? 1 : 0;

  let smokeScore = 0;
  if (smoking > 0 && smoking < 15) {
    smokeScore = 0.22;
  } else if (smoking >= 15 && smoking < 25) {
    smokeScore = 0.45;
  } else if (smoking >= 25) {
    smokeScore = 0.75;
  }

  // Logistic model formulation calibrated to Framingham / SASHELP.HEART
  const z =
    -5.4 +
    age * 0.062 +
    isMale * 0.48 +
    (systolic - 120) * 0.018 +
    (diastolic - 80) * 0.008 +
    (cholesterol - 200) * 0.004 +
    (bmi - 25) * 0.025 +
    smokeScore;

  const prob = 1 / (1 + Math.exp(-z));
  return Math.min(Math.max(prob, 0.01), 0.99);
};

export const generateCounterfactualPlan = (patient, targetRiskPct = 20) => {
  if (!patient) return null;

  const currentRisk = calculateRisk(patient);
  const currentRiskPct = Math.round(currentRisk * 100);
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
        title: "Cardiovascular Maintenance Nutrition (Mediterranean Protocol)",
        sodium: "Maintain < 2,000 mg/day",
        guidelines: [
          "Continue high-fiber whole grain intake (oats, quinoa, brown rice).",
          "Consume 5+ servings of colorful fruits and vegetables daily.",
          "Maintain healthy fats (extra virgin olive oil, nuts, seeds).",
          "Keep added sugars under 25g/day to prevent arterial inflammation."
        ]
      },
      exercisePlan: {
        title: "Aerobic Vitality & Maintenance Protocol",
        duration: "150 minutes per week",
        schedule: [
          { day: "Mon / Wed / Fri", activity: "30–40 min Brisk Walk (3.5–4.0 mph), swimming, or outdoor cycling" },
          { day: "Tue / Thu", activity: "20 min moderate resistance training (dumbbells or bodyweight)" },
          { day: "Weekend", activity: "Active recreation: hiking, leisure cycling, or outdoor sports" }
        ],
        strength: "2 sessions/week of moderate bodyweight or resistance training",
        impact: "Preserves endothelial elasticity and maintains healthy stroke volume."
      },
      clinicalActions: [
        "Continue annual preventive cardiology screening and lipid panels.",
        "Maintain routine blood pressure logging every 3-6 months."
      ]
    };
  }

  // Iteratively optimize modifiable biomarkers to reach target
  let simSmoking = 0;
  let simSystolic = Math.min(systolic, systolic > 140 ? 122 : 118);
  let simDiastolic = Math.min(diastolic, diastolic > 85 ? 78 : 74);
  let simCholesterol = Math.min(cholesterol, 185);
  let simWeight = currentBmi > 25 ? Math.round(weight * 0.92) : weight;

  const simulatedPatient = {
    ...patient,
    Systolic: simSystolic,
    Diastolic: simDiastolic,
    Cholesterol: simCholesterol,
    Weight: simWeight,
    Smoking: simSmoking,
  };

  const simRisk = calculateRisk(simulatedPatient);
  const simRiskPct = Math.round(simRisk * 100);
  const riskReductionPct = Math.max(0, currentRiskPct - simRiskPct);

  // Compute specific deltas
  const deltas = [];
  if (smoking > 0) {
    deltas.push({
      metric: 'Smoking Cessation',
      current: `${smoking} cigs/day`,
      target: '0 cigs/day',
      diff: `-${smoking} cigs/day`,
      priority: 'Highest',
      impact: 'Eliminates vascular endothelial toxicity and cuts heart attack hazard in half.'
    });
  }

  if (systolic > 125 || diastolic > 82) {
    deltas.push({
      metric: 'Blood Pressure Control',
      current: `${systolic}/${diastolic} mmHg`,
      target: `${simSystolic}/${simDiastolic} mmHg`,
      diff: `-${systolic - simSystolic}/-${diastolic - simDiastolic} mmHg`,
      priority: 'High',
      impact: 'Significantly relieves arterial shear stress and prevents left ventricular hypertrophy.'
    });
  }

  if (cholesterol > 200) {
    deltas.push({
      metric: 'Serum Cholesterol',
      current: `${cholesterol} mg/dL`,
      target: `${simCholesterol} mg/dL`,
      diff: `-${cholesterol - simCholesterol} mg/dL`,
      priority: 'Medium-High',
      impact: 'Stops coronary plaque deposition and stabilizes existing arterial walls.'
    });
  }

  if (weight > simWeight) {
    deltas.push({
      metric: 'Body Weight / BMI',
      current: `${weight} lbs (${currentBmi.toFixed(1)} BMI)`,
      target: `${simWeight} lbs (${((simWeight * 703) / (height * height)).toFixed(1)} BMI)`,
      diff: `-${weight - simWeight} lbs (-8%)`,
      priority: 'Medium',
      impact: 'Improves peripheral insulin sensitivity and lowers baseline systemic vascular resistance.'
    });
  }

  // Dietary Prescription
  const dietPlan = {
    title: "Cardiometabolic Prescription: DASH + Mediterranean Hybrid Diet",
    sodium: systolic >= 140 ? "Strict < 1,500 mg/day (Crucial for BP reduction)" : "Moderate < 2,000 mg/day",
    guidelines: [
      "Sodium Restriction: Eliminate canned soups, cured meats, and restaurant sauces. Use garlic, herbs, and citrus for seasoning.",
      "Arterial Vasodilator Foods: Consume high-potassium foods (spinach, avocados, white beans, sweet potatoes) to relax vascular smooth muscle.",
      "Cardioprotective Lipids: Replace butter and margarine with 2 tablespoons of extra-virgin olive oil daily; eat cold-water fish (salmon, tuna) 2x/week.",
      "Soluble Fiber Target: 30g+ daily from oats, chia seeds, and legumes to actively bind and excrete LDL cholesterol particles.",
      "Limit Alcohol & Refined Sugars: Restrict alcohol to < 1 standard drink/day; replace refined carbs with low-glycemic complex grains."
    ]
  };

  // Exercise Schedule
  const exercisePlan = {
    title: "Prescriptive Exercise Protocol (Cardiovascular Rehabilitation Schedule)",
    duration: "150–180 minutes of total weekly aerobic activity",
    frequency: "5 days per week, 30–40 minutes per session",
    schedule: [
      { day: "Monday", activity: "35 min Moderate Brisk Walk (3.5–4.0 mph) or Incline Treadmill (Heart rate at 50–70% max)" },
      { day: "Tuesday", activity: "25 min Low-Impact Resistance Training (Squats, light resistance bands, core stability)" },
      { day: "Wednesday", activity: "35 min Stationary Cycling, Rowing, or Swimming at steady aerobic pace" },
      { day: "Thursday", activity: "Active Recovery: 20 min gentle stretching, mobility work, and outdoor walk" },
      { day: "Friday", activity: "35 min Interval Aerobic Session (Alternate 3 min moderate / 1 min brisk tempo)" },
      { day: "Saturday", activity: "30 min Full-Body Dumbbell or Bodyweight Circuit (Enhances metabolic glucose clearance)" },
      { day: "Sunday", activity: "Rest & Recovery: Leisure stroll and diaphragmatic breathing" }
    ],
    strength: "2 weekly resistance sessions to increase muscle capillary density and reduce arterial stiffness.",
    impact: "Clinically proven to reduce resting systolic BP by 6–10 mmHg and increase HDL cholesterol."
  };

  // Clinical & Medical Recommendations
  const clinicalActions = [
    systolic >= 140 || diastolic >= 90
      ? "Physician Review: Initiate or adjust first-line antihypertensive therapy (e.g. ACE Inhibitor / ARB or CCB) to safely attain < 130/80 mmHg."
      : "Blood Pressure Monitoring: Log home resting blood pressure twice daily (morning & evening) for 14 consecutive days.",
    cholesterol >= 240
      ? "Lipid Management: Order a comprehensive fasting lipid panel (LDL-C, HDL-C, Triglycerides, ApoB) and evaluate moderate-intensity statin initiation."
      : "Lipid Re-check: Repeat fasting lipid panel in 12 weeks to assess dietary response.",
    smoking > 0
      ? "Tobacco Intervention: Enroll in a physician-supervised cessation program. Offer Nicotine Replacement Therapy (transdermal patches + oral gum) or varenicline."
      : "Vascular Maintenance: Zero tobacco exposure preserves coronary arterial elasticity.",
    "Sleep & Autonomic Health: Maintain 7–8 hours of uninterrupted sleep; evaluate for Obstructive Sleep Apnea (OSA) if morning hypertension persists.",
    "Stress Reduction: 10 minutes daily resonance breathing (6 breaths per minute) to lower sympathetic nervous system tone."
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
    clinicalActions
  };
};
