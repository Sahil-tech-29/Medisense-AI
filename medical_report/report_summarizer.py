import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def summarize_medical_report(report_text):

    prompt = f"""
You are a medical awareness assistant.

TASK:
Summarize the following medical report in simple, easy-to-understand language.

STRICT RULES:
- Do NOT diagnose diseases
- Do NOT prescribe medicines or treatments
- Do NOT provide medical thresholds or numeric interpretation
- Do NOT give emergency instructions
- Do NOT replace a healthcare professional

YOU MAY:
- Explain what the report is generally about
- Highlight key findings in a high-level way
- Explain common medical terms
- Encourage consulting a healthcare professional

FORMAT:
1. Report Overview
2. Key Observations
3. General Understanding
4. When to Consult a Professional
5. Disclaimer

MEDICAL REPORT:
{report_text}
"""

    response = model.generate_content(prompt)
    return response.text


# medical_report/report_summarizer.py

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # 👈 loads .env file

