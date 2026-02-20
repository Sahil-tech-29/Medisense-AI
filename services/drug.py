from utils.groq_client import generate_with_groq
from utils.helpers import confidence_band

CRITICAL_TERMS = [
    "death", "died", "fatal", "life threatening",
    "chest pain", "heart pain", "heart issue",
    "difficulty breathing", "breathless", "shortness of breath",
    "collapsed", "unconscious", "coma",
    "seizure", "convulsion", "fits",
    "stroke", "paralysis",
    "severe bleeding", "vomiting blood",
    "anaphylaxis","death", "died", "fatal", "life threatening",
    "chest pain", "heart pain", "breathless", "shortness of breath",
    "bleeding", "seizure", "convulsion",
    "stroke", "heart attack",
    "organ failure", "coma", "unconscious",
    "paralysis", "anaphylaxis"
]

ALLERGIC_TERMS = [
    "rash", "hives", "itching", "swelling",
    "face swelling", "lip swelling", "throat swelling",
    "skin redness", "burning skin",
    "allergy", "allergic reaction"
]

CONCERNING_TERMS = [
    "persistent pain", "severe pain","vomit","vomiting"
    "vomiting", "continuous vomiting",
    "dizziness", "faint",
    "high fever", "fever",
    "confusion", "blurred vision",
    "palpitations", "irregular heartbeat",
    "weakness", "numbness"
]
MILD_EFFECTS = [
    "nausea", "mild headache",
    "fatigue", "sleepy",
    "dry mouth", "constipation",
    "diarrhea", "loss of appetite"
]


def extract_side_effects(text):
    return list({effect for effect in MILD_EFFECTS if effect in text})


def analyze_drug_review(review):
    text = review.lower()

    critical = any(term in text for term in CRITICAL_TERMS)
    allergic = any(term in text for term in ALLERGIC_TERMS)
    concerning = any(term in text for term in CONCERNING_TERMS)
    mild = extract_side_effects(text)

    # 🔴 HARD STOP
    if critical:
        return {
            "risk_label": "High Risk",
            "confidence": 96.0,
            "effects": mild,
            "reason": "Life-threatening or emergency indicators detected"
        }

    # 🟠 Allergy warning
    if allergic:
        return {
            "risk_label": "Moderate to High Risk",
            "confidence": 85.0,
            "effects": mild,
            "reason": "Possible allergic reaction detected"
        }

    # 🟡 Concerning symptoms
    if concerning:
        return {
            "risk_label": "Moderate Risk",
            "confidence": 78.0,
            "effects": mild,
            "reason": "Symptoms may require medical attention"
        }

    # 🟢 Mild-only
    if mild:
        return {
            "risk_label": "Likely Safe",
            "confidence": 70.0,
            "effects": mild,
            "reason": "Only commonly reported mild side effects detected"
        }

    # ⚪ Unknown
    return {
        "risk_label": "Uncertain",
        "confidence": 55.0,
        "effects": [],
        "reason": "Insufficient or unclear information"
    }


def generate_drug_risk_explanation(review, risk_label, confidence, effects):
    effects_list = ", ".join(effects) if effects else "No clearly mentioned side effects"

    prompt = f"""
You are a healthcare awareness assistant.

Generate STRICT HTML output.

Write detailed, multi-sentence explanations for EACH section.
Do NOT summarize too briefly.

Allowed tags ONLY:
<h4>, <p>, <ul>, <li>

Follow this exact structure and WRITE DETAILED CONTENT for EACH section.

<h4>Side effects mentioned</h4>
<ul>
<li>Clearly list and briefly explain each side effect mentioned by the user</li>
</ul>

<h4>What this risk level means</h4>
<p>
Explain what "{risk_label}" means IN CONTEXT of this specific review.
Explain how AI generally interprets such experiences.
</p>

<h4>Why these effects may be manageable or concerning</h4>
<ul>
<li>Explain how intensity, duration, and combination of effects matter</li>
<li>Explain why some effects are usually mild while others need attention</li>
</ul>

<h4>Safety awareness (non-medical)</h4>
<ul>
<li>What to observe in the body</li>
<li>What changes should be noted</li>
<li>Why awareness matters</li>
</ul>

<h4>When to consider professional help</h4>
<ul>
<li>Clear signs that should not be ignored</li>
<li>Situations where medical advice is important</li>
</ul>

IMPORTANT:
If severe symptoms are present, treat the case as high risk
even if symptoms are rare or unclear.


Rules:
- Do NOT diagnose
- Do NOT prescribe medicines
- Do NOT introduce new side effects
- Use calm, professional, reassuring language
- Avoid fear-based language
- Output ONLY valid HTML

User review:
{review}
"""

    return generate_with_groq(prompt)



