def build_preventive_prompt(risk_results):
    risk_summary = ""
    for risk, data in risk_results.items():
        risk_summary += f"- {risk}: {data['level']} risk\n"

    prompt = f"""
You are a healthcare awareness assistant.

Based on the following preventive health risk tendencies:
{risk_summary}

Provide:
- Simple explanation of what these risks mean
- How lifestyle factors influence them
- Preventive awareness tips (non-medical)
- Early warning signs (non-clinical)
- When professional medical advice may be helpful

Rules:
- Do NOT diagnose diseases
- Do NOT recommend medicines or treatments
- Keep language calm, supportive, and educational
"""

    return prompt
