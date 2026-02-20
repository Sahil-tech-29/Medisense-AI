from preventive_health.risk_engine import calculate_health_risks
from utils.groq_client import generate_with_groq


def build_preventive_prompt(risk_results, user_inputs):
    risk_summary = ""
    for risk, data in risk_results.items():
        risk_summary += f"- {risk}: {data['level']} risk\n"

    prompt = f"""
You are a preventive health awareness assistant.

Generate STRICT HTML output.

Allowed tags ONLY:
<h4>, <p>, <ul>, <li>

Follow this EXACT structure and write meaningful, detailed content
for EACH section. Avoid long paragraphs.

<h4>Overall preventive health summary</h4>
<p>
Explain what the overall risk profile suggests about the user's
current preventive health status in simple terms.
</p>

<h4>What these risk areas represent</h4>
<ul>
<li>Metabolic Health: blood sugar, cholesterol, and weight balance</li>
<li>Cardiovascular Lifestyle: heart health and daily habits</li>
<li>Mental Well-being: stress handling and emotional balance</li>
<li>Sleep & Fatigue: rest quality, recovery, and daily energy</li>
</ul>

<h4>Positive signs observed</h4>
<ul>
<li>Mention areas where risk is low or habits appear supportive</li>
</ul>

<h4>Areas to be mindful about</h4>
<ul>
<li>Explain habits that may increase future risk if unchanged</li>
</ul>

<h4>Preventive lifestyle guidance (non-medical)</h4>
<ul>
<li>Simple daily habits the user can realistically follow</li>
<li>Focus on prevention and consistency</li>
</ul>

<h4>Early signs that should not be ignored</h4>
<ul>
<li>General warning signs related to energy, stress, sleep, or heart health</li>
</ul>

Rules:
- DO NOT diagnose
- DO NOT predict disease
- DO NOT prescribe medicine
- Use calm, encouraging, human language
- Output ONLY valid HTML

User inputs:
{user_inputs}

Risk analysis:
{risk_summary}
"""
    return prompt


def assess_preventive_health(user_inputs):
    risks = calculate_health_risks(user_inputs)

    prompt = build_preventive_prompt(risks, user_inputs)
    guidance = generate_with_groq(prompt)

    return {
        "risks": risks,
        "guidance": guidance
    }
