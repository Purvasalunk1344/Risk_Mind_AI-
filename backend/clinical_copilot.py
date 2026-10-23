"""
Clinical AI Copilot Engine - Heart Disease Risk Studio
======================================================
Supports multiple FREE LLM providers. Set your API key in backend/.env:

  GROQ_API_KEY=gsk_...       (FREE - Llama 3.3 70B, fastest)
  GEMINI_API_KEY=AIza...     (FREE - Gemini 1.5 Flash)
  OPENAI_API_KEY=sk-...      (Paid, optional fallback)

Get free keys:
  Groq  -> https://console.groq.com   (completely free, no credit card)
  Gemini-> https://aistudio.google.com/app/apikey (free tier)
"""

import os
from pathlib import Path

# Load .env from backend/ directory
def _load_dotenv():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    val = v.strip().strip('"').strip("'")
                    if val:
                        os.environ[k.strip()] = val

_load_dotenv()


def compute_derived_metrics(p):
    """Compute all clinical metrics from raw patient dict."""
    age = float(p.get("AgeAtStart", 50) or 50)
    sex = str(p.get("Sex", "Female")).capitalize()
    systolic = float(p.get("Systolic", 120) or 120)
    diastolic = float(p.get("Diastolic", 80) or 80)
    chol = float(p.get("Cholesterol", 200) or 200)
    weight = float(p.get("Weight", 160) or 160)
    height = float(p.get("Height", 66) or 66)
    smoking = float(p.get("Smoking", 0) or 0)
    risk = float(p.get("risk", 0.20) or 0.20)

    pulse_pressure = int(systolic - diastolic)
    map_bp = round(diastolic + (systolic - diastolic) / 3.0, 1)
    bmi = round((weight * 703.0 / (height ** 2)) if height > 0 else 25.0, 1)

    if systolic < 120 and diastolic < 80:
        bp_stage = "Normal"
    elif systolic <= 129 and diastolic < 80:
        bp_stage = "Elevated"
    elif systolic <= 139 or diastolic <= 89:
        bp_stage = "Stage 1 Hypertension"
    elif systolic <= 180 or diastolic <= 120:
        bp_stage = "Stage 2 Hypertension"
    else:
        bp_stage = "Hypertensive Crisis"

    pp_flag = "Elevated >60 mmHg (High Arterial Stiffness)" if pulse_pressure > 60 else \
              "Narrow <30 mmHg (Low Stroke Volume)" if pulse_pressure < 30 else \
              "Optimal (40-60 mmHg)"
    chol_flag = "Desirable <200 mg/dL" if chol < 200 else \
                "Borderline High 200-239 mg/dL" if chol < 240 else \
                "High Hypercholesterolemia >=240 mg/dL"
    bmi_flag = "Underweight" if bmi < 18.5 else \
               "Healthy Weight" if bmi < 25 else \
               "Overweight" if bmi < 30 else "Obese"
    smoke_flag = "Non-smoker" if smoking == 0 else \
                 f"Light ({int(smoking)} cigs/day)" if smoking < 15 else \
                 f"Moderate ({int(smoking)} cigs/day)" if smoking < 25 else \
                 f"Heavy ({int(smoking)}+ cigs/day)"

    return {
        "age": int(age), "sex": sex,
        "systolic": int(systolic), "diastolic": int(diastolic),
        "pulse_pressure": pulse_pressure, "map_bp": map_bp,
        "chol": int(chol), "weight": int(weight), "height": int(height),
        "smoking": int(smoking), "bmi": bmi,
        "bp_stage": bp_stage, "pp_flag": pp_flag,
        "chol_flag": chol_flag, "bmi_flag": bmi_flag, "smoke_flag": smoke_flag,
        "risk_pct": round(risk * 100, 1),
        "risk_band": "Low" if risk < 0.20 else "Medium" if risk < 0.50 else "High"
    }


def _build_system_prompt(m):
    """Build the rich clinical context system prompt sent to any LLM."""
    return f"""You are **HeartRisk Copilot**, an elite Clinical Cardiology AI assistant embedded in the HeartRisk Studio application.

Your primary user is a **licensed attending physician or cardiologist**. Always respond with:
- Rigorous clinical terminology (ACC/AHA guidelines, hemodynamics, GDMT, pharmacology)
- Evidence-based recommendations with drug names, dosages, and trial citations where relevant
- Structured markdown formatting (headers, bullet points, bold for key values)
- Concise yet comprehensive clinical depth — no fluff, no disclaimers

== CURRENT PATIENT CLINICAL CONTEXT ==
Patient Name    : {m['name']}
Age / Sex       : {m['age']} years / {m['sex']}
Blood Pressure  : {m['systolic']}/{m['diastolic']} mmHg ({m['bp_stage']})
Pulse Pressure  : {m['pulse_pressure']} mmHg ({m['pp_flag']})
Mean Art. Press : {m['map_bp']} mmHg
Total Cholesterol: {m['chol']} mg/dL ({m['chol_flag']})
BMI             : {m['bmi']} ({m['bmi_flag']})
Tobacco Exposure: {m['smoking']} cigarettes/day ({m['smoke_flag']})
10-Year Mortality Risk: {m['risk_pct']}% ({m['risk_band']} Tier) — Framingham ML model

Always tailor your answers to THIS specific patient's values. If the doctor asks a general question, still contextualize it to this patient profile. Be direct, precise, and actionable."""


def _call_groq(system_prompt, messages, model="llama-3.3-70b-versatile"):
    """Call Groq API (FREE - Llama 3.3 70B). Get key at https://console.groq.com"""
    import groq
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt, messages):
    """Call Google Gemini API (FREE tier). Get key at https://aistudio.google.com/app/apikey"""
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    candidate_models = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-1.5-flash"]
    last_err = None

    gemini_history = []
    user_message = messages[-1]["content"] if messages else ""
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(user_message)
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            continue
    raise last_err or RuntimeError("No Gemini model succeeded")


def _call_openai(system_prompt, messages, model="gpt-3.5-turbo"):
    """Call OpenAI API (optional, paid)."""
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    response = openai.ChatCompletion.create(
        model=model,
        messages=full_messages,
        temperature=0.2,
        max_tokens=900,
    )
    return response.choices[0].message["content"]


def _get_provider():
    """Returns the first available LLM provider based on configured API keys."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    return None


def answer_clinical_query(message, patient, history=None):
    """
    Main Clinical Copilot entry point.
    Automatically selects the best available LLM provider from .env keys.
    Falls back to an embedded clinical knowledge engine if no keys are configured.
    """
    metrics = compute_derived_metrics(patient)
    metrics["name"] = patient.get("name") or "Current Patient"

    system_prompt = _build_system_prompt(metrics)

    # Build conversation history in OpenAI format (all providers accept this format)
    conv_messages = []
    if history and isinstance(history, list):
        for h in history[-6:]:  # Keep last 6 turns of context
            role = h.get("role", "user")
            content = h.get("content", "")
            if role in ("user", "assistant") and content:
                conv_messages.append({"role": role, "content": content})
    conv_messages.append({"role": "user", "content": message})

    provider = _get_provider()

    try:
        if provider == "groq":
            return _call_groq(system_prompt, conv_messages)
        elif provider == "gemini":
            return _call_gemini(system_prompt, conv_messages)
        elif provider == "openai":
            return _call_openai(system_prompt, conv_messages)
    except Exception as e:
        print(f"[WARN] LLM provider '{provider}' failed: {e}. Falling back to embedded engine.")

    # --- Embedded offline fallback (no API key needed) ---
    return _embedded_fallback(message, metrics)


def _embedded_fallback(query, m):
    """High-quality embedded clinical knowledge engine — works fully offline."""
    q = query.lower()
    name = m["name"]
    no_key_notice = (
        "\n\n---\n> **Note:** No LLM API key detected. "
        "Add a **free** `GROQ_API_KEY` or `GEMINI_API_KEY` to `backend/.env` "
        "for full AI-powered conversational responses. "
        "See [console.groq.com](https://console.groq.com) (free, no credit card)."
    )

    if any(k in q for k in ["diet", "food", "sodium", "dash", "salt", "nutrition", "lower systolic"]):
        return f"""### Dietary Guidance: Lowering Systolic BP

**Patient:** {name} | BP: `{m['systolic']}/{m['diastolic']} mmHg` ({m['bp_stage']})

**DASH Diet** (fastest non-pharmacologic option):
- Expected drop: **8–14 mmHg** within 2–4 weeks
- High potassium, magnesium, calcium — limits saturated fat and sodium

**Sodium Restriction:**
- Target: **< 1,500 mg/day** (from typical ~3,400 mg/day)
- Combined with DASH: up to **15–21 mmHg** drop (comparable to monotherapy)

**Additional evidence-based interventions:**
- Beetroot juice / nitrate-rich foods: acute 4–10 mmHg drop
- Weight loss: ~1 mmHg per 1 kg lost (BMI currently `{m['bmi']} — {m['bmi_flag']}`)
- Physical activity: 150 min/week aerobic → 5–8 mmHg drop
- Alcohol restriction: <2 drinks/day → 3–4 mmHg drop
{no_key_notice}"""

    if any(k in q for k in ["pulse pressure", "yellow", "amber", "flagged", "stiffness"]):
        pp = m["pulse_pressure"]
        if pp > 60:
            status = f"ELEVATED at {pp} mmHg (>60 mmHg threshold — flagged amber)"
        elif pp < 30:
            status = f"NARROW at {pp} mmHg (<30 mmHg — low stroke volume)"
        else:
            status = f"OPTIMAL at {pp} mmHg (40–60 mmHg range)"
        return f"""### Pulse Pressure Analysis

**Patient:** {name} | BP: `{m['systolic']}/{m['diastolic']} mmHg`  
**Pulse Pressure = SBP - DBP = {pp} mmHg** — Status: **{status}**

**Why it gets flagged amber/yellow (PP > 60 mmHg):**
Elevated PP indicates **reduced aortic compliance** (vascular stiffening). The elastic Windkessel effect is impaired — the aortic wall no longer dampens pulsatile flow. Reflected pressure waves arrive in **late systole** instead of diastole, amplifying SBP while lowering DBP. This increases:
- LV afterload and myocardial O2 demand
- Pulsatile stress on cerebral microvasculature (stroke risk)
- Renal glomerular hyperperfusion (nephrosclerosis risk)

**Clinical Action:**
- Preferred agents: **Amlodipine 5–10 mg** (CCB) and ACEi/ARB — these reduce central aortic stiffness
- Avoid: Beta-blockers as monotherapy (increase peripheral wave reflection)
- Target SBP < 120 mmHg while maintaining DBP > 65 mmHg
{no_key_notice}"""

    if any(k in q for k in ["medication", "drug", "prescribe", "statin", "ace", "arb", "ccb"]):
        dual_rec = ("**Dual therapy recommended** (BP >20/10 above goal)"
                    if m['systolic'] >= 150 or m['diastolic'] >= 90 else "Monotherapy appropriate")
        statin = ("**High-intensity statin** (Atorvastatin 40–80 mg)"
                  if m['risk_pct'] >= 20 or m['chol'] >= 240 else "**Moderate-intensity statin** (Atorvastatin 10–20 mg)")
        return f"""### GDMT Pharmacotherapy Recommendations

**Patient:** {name} | Risk: `{m['risk_pct']}%` | BP: `{m['systolic']}/{m['diastolic']} mmHg`

**Antihypertensive:** {dual_rec}
- Option A: ACEi (Lisinopril 10–20 mg QD) or ARB (Losartan 50–100 mg QD) + CCB (Amlodipine 5 mg QD)
- Option B: ARB + Thiazide-like diuretic (Chlorthalidone 12.5–25 mg QD)

**Lipid-lowering:** {statin}
- Cholesterol `{m['chol']} mg/dL` ({m['chol_flag']})
- Goal: LDL-C reduction ≥50% from baseline

{"**Smoking cessation:** Varenicline (Chantix) 0.5 mg QD → 1 mg BID (highest efficacy); OR Bupropion SR 150 mg BID" if m['smoking'] > 0 else "**Smoking:** Patient is non-smoker — no cessation pharmacotherapy needed."}

**Follow-up:** BMP in 2–4 weeks post-ACEi/ARB initiation.
{no_key_notice}"""

    if any(k in q for k in ["smoke", "smoking", "tobacco", "cigarette", "nicotine"]):
        return f"""### Tobacco Exposure & Cardiovascular Risk

**Patient:** {name} | Exposure: `{m['smoking']} cigs/day` ({m['smoke_flag']})

**Quantitative Framingham Impact:**
- Moderate-heavy smoking multiplies 10-year mortality odds by **2.2x–2.7x**
- Estimated contribution to this patient's risk: **+8% to +14%** absolute

**Cessation Timeline Benefits:**
| Timeframe | Benefit |
|-----------|---------|
| 24 hours | CO normalizes, O2 saturation improves |
| 1 month | SBP drops 3–6 mmHg (adrenergic vasoconstriction reversal) |
| 1 year | CV mortality excess risk drops **50%** |
| 5–10 years | Risk approaches that of a lifetime non-smoker |

**Prescribe (5 A's framework):** Varenicline 0.5 mg QD × 7 days → 1 mg BID × 12 weeks (highest quit rate)
{no_key_notice}"""

    # General fallback
    return f"""### Clinical Consultation — {name}

**Vitals:** BP `{m['systolic']}/{m['diastolic']} mmHg` ({m['bp_stage']}) | PP `{m['pulse_pressure']} mmHg` | Cholesterol `{m['chol']} mg/dL` | BMI `{m['bmi']}` | Smoking `{m['smoking']} cigs/day`  
**10-Year Framingham Mortality Risk: `{m['risk_pct']}%` ({m['risk_band']} Tier)**

**Regarding your query:** *"{query}"*

Based on this patient's profile, key clinical priorities are:
1. **Hemodynamic control** — Target BP < 130/80 mmHg (preferably <120/80 mmHg)
2. **Lipid management** — Statin therapy indicated (risk {m['risk_pct']}%, Chol {m['chol']} mg/dL)
3. **Lifestyle prescription** — DASH diet, 150 min/week aerobic exercise, sodium <1,500 mg/day
4. **Reassessment** — BMP + fasting lipids in 4–8 weeks
{no_key_notice}"""
