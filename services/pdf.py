from medical_report.pdf_extraction import extract_text_from_pdf, is_readable_text
from utils.groq_client import generate_with_groq


def summarize_medical_report(report_text):
    prompt = f"""
You are a medical awareness assistant.

Generate STRICT HTML output.

Allowed tags ONLY:
<h4>, <p>, <ul>, <li>

Follow this exact structure:

<h4>Report Overview</h4>
<p>High-level explanation of what this report is about.</p>

<h4>Key Observations</h4>
<ul>
<li>Only important findings</li>
</ul>

<h4>General Understanding</h4>
<p>Explain medical terms simply.</p>

<h4>When to Consult a Professional</h4>
<ul>
<li>Situations where medical advice is important</li>
</ul>

<h4>Disclaimer</h4>
<p>This is for awareness only.</p>

Report:
{report_text}
"""
    return generate_with_groq(prompt)


def process_medical_pdf(uploaded_file):
    text = extract_text_from_pdf(uploaded_file)

    if not is_readable_text(text):
        return {
            "error": "Scanned or image-based PDF detected. OCR required."
        }

    if not is_medical_report(text):
        return {
            "error": "The uploaded file does not appear to be a medical report."
        }
    
    summary = summarize_medical_report(text)

    return {
        "text": text,
        "summary": summary
    }


MEDICAL_KEYWORDS = [
    "patient", "diagnosis", "test", "cholesterol", "blood",
    "report", "medical", "mg/dl", "reference range",
    "clinical", "lab", "result", "profile", 
    "patient", "age", "gender", "diagnosis", "test",
    "report", "result", "mg/dl", "mmol", "blood",
    "cholesterol", "hdl", "ldl", "triglyceride",
    "hemoglobin", "cbc", "lipid", "thyroid",
    "urine", "glucose", "bp", "clinical"
]

def is_medical_report(text):
    text = text.lower()
    matches = sum(keyword in text for keyword in MEDICAL_KEYWORDS)
    return matches >= 3  
