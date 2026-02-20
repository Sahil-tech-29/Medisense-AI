from utils.groq_client import generate_with_groq


def generate_mental_health_explanation(text, label):
    prompt = f"""
You are a mental health awareness assistant.

Detected emotional pattern: {label}

User text:
{text}

Generate the response in STRICT HTML format.

Allowed tags ONLY:
<h4>, <p>, <ul>, <li>

You MUST follow this exact structure:

<h4>What this emotional state generally means</h4>
<p>
Explain the emotional state in simple, non-clinical language.
Avoid labels, diagnoses, or medical terms.
</p>

<h4>Why these feelings may be occurring</h4>
<ul>
<li>Life stressors or pressure</li>
<li>Emotional overload</li>
<li>Situational or environmental factors</li>
</ul>

<h4>How this state commonly affects people</h4>
<p>
Explain emotional, mental, or physical impact on daily life.
</p>

<h4>Gentle self-care and coping suggestions</h4>
<ul>
<li>Breathing and grounding techniques</li>
<li>Rest, routine, and small supportive habits</li>
</ul>

<h4>When it may help to talk to someone</h4>
<ul>
<li>Feelings persist or worsen</li>
<li>Daily functioning becomes difficult</li>
<li>You feel overwhelmed or stuck</li>
</ul>

Rules:
- DO NOT diagnose
- DO NOT label the user with a disorder
- DO NOT prescribe medication
- Avoid alarming or crisis language
- Keep tone calm, empathetic, and human
- Output ONLY valid HTML
"""
    return generate_with_groq(prompt)
