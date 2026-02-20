import re, string
from utils.groq_client import generate_with_groq
from utils.helpers import confidence_band


def ai_predict_disease(symptoms):
    prompt = f"""
    You are an intelligent medical symptom analysis system.

    User symptoms:
    "{symptoms}"

    TASK:
    - Identify the SINGLE most likely condition.
    - If symptoms are very mild or vague, return "General Viral Infection".
    - Return ONLY the disease name.
    - Do NOT explain.
    - Do NOT add extra text.
    - Output must be a short disease name.

    Examples:
    - "mild cold and cough" → Common Cold
    - "fever body pain headache" → Viral Fever
    - "chest pain sweating breathlessness" → Possible Heart Condition

    Return only the disease name.
    """

    response = generate_with_groq(prompt)
    return response.strip()


def clean_symptom_input(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_valid_symptom_input(text):
    words = text.split()
    
    # too short
    if len(words) < 2:
        return False

    # contains at least one alphabetic word
    valid_words = [w for w in words if w.isalpha() and len(w) > 2]

    return len(valid_words) >= 2


def ai_explain_disease(symptoms, disease):
    prompt = f"""
You are a healthcare awareness assistant.

Condition identified: {disease}
User symptoms: {symptoms}

Generate a detailed but clear explanation in STRICT HTML.

ALLOWED TAGS ONLY:
<h4>, <p>, <ul>, <li>

STRUCTURE (FOLLOW EXACTLY):

<h4>What this condition generally means</h4>
<p>
Explain the condition clearly, include what part of the body it affects
and why it commonly occurs. Avoid medical jargon.
</p>

<h4>Why these symptoms match</h4>
<ul>
<li>Explain each symptom in relation to the condition</li>
<li>Use simple cause-and-effect language</li>
</ul>

<h4>How this condition typically progresses</h4>
<p>
Explain whether it usually improves on its own, stays stable,
or may worsen without care.
</p>

<h4>Severity (usually mild or serious)</h4>
<p>
Explain typical severity and reassurance without dismissing concerns.
</p>

<h4>General self-care and precautions (non-medical)</h4>
<ul>
<li>Rest, posture, hydration, lifestyle guidance</li>
<li>Daily habits that may reduce discomfort</li>
</ul>

<h4>When to consult a doctor</h4>
<ul>
<li>Symptoms lasting longer than expected</li>
<li>New or worsening symptoms</li>
<li>Symptoms affecting daily activities</li>
</ul>

RULES:
- Educational tone only
- No diagnosis
- No medicines
- No emergency language
- Output ONLY valid HTML
"""
    return generate_with_groq(prompt)
