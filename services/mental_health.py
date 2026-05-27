import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Load saved model 
model = load_model("model/mental_health/mental_model.h5" ,compile = False)

with open("model/mental_health/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/mental_health/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Keeping the max length same as in ipynb file 
MAX_LEN = 120


# Clean the text using NLP

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# LSTM prediction - Return top-3 LSTM predictions with normalized confidence.

def predict_top3(text: str) -> list:
    text = clean_text(text)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)

    probs = model.predict(pad, verbose=0)[0]

    total = np.sum(probs)
    if total == 0:
        return []
    probs = probs / total

    top3_idx = np.argsort(probs)[-3:][::-1]

    results = [
        {
            "label": le.inverse_transform([i])[0],
            "confidence": float(probs[i])
        }
        for i in top3_idx
    ]

    return results

#  Groq LLM verification 
# Use Groq LLaMA-3 to choose the most accurate diagnosis from LSTM top-3.Returns the disease name as a plain string.
def verify_with_llm(text: str, top3: list) -> str:
    from utils.groq_client import generate_with_groq

    options = "\n".join([f"- {d['label']}" for d in top3])

    prompt = f"""
User text:
{text}

Predicted emotional states:
{options}

Select the MOST appropriate emotional state.

Rules:
- Choose ONLY from given options
- Do NOT add new labels
- Do NOT diagnose or use clinical terms
- Be conservative and realistic
- Return ONLY one label
"""

    return generate_with_groq(prompt).strip()

from utils.groq_client import generate_with_groq

def generate_mental_health_explanation(text, label):
    prompt = f"""
You are a mental health awareness assistant.

Detected emotional state: {label}

User text:
{text}

Generate response in STRICT HTML format.

Allowed tags:
<h4>, <p>, <ul>, <li>

<h4>What this emotional state means</h4>
<p>Explain in simple, non-clinical language.</p>

<h4>Why this may happen</h4>
<ul>
<li>Stress or pressure</li>
<li>Emotional overload</li>
<li>Daily life factors</li>
</ul>

<h4>How it affects daily life</h4>
<p>Explain impact.</p>

<h4>Helpful suggestions</h4>
<ul>
<li>Rest</li>
<li>Talk to someone</li>
<li>Take breaks</li>
</ul>

Rules:
- No diagnosis
- No medication
- Calm tone
- Output ONLY HTML
"""
    return generate_with_groq(prompt)



# if having following keywords in the text provided by user 
def filter_predictions(top3, text):
    text = text.lower()

    def boost(label_name):
        matched = [d for d in top3 if d["label"] == label_name]
        others = [d for d in top3 if d["label"] != label_name]
        return matched + others if matched else top3

    # 🟢 POSITIVE / NORMAL STATE
    if any(w in text for w in [
        "happy", "fine", "good", "relaxed", "okay", "enjoy", "normal", "satisfied"
    ]):
        return [d for d in top3 if d["label"] == "Generally Stable"] or top3
    
    
    # Anxiety
    if any(w in text for w in [
        "overthinking", "worry", "worried", "nervous", "anxious", "fear", "panic"
    ]):
        return boost("Anxiety Patterns")

    # Stress
    if any(w in text for w in [
        "pressure", "deadline", "workload", "busy", "overwhelmed", "too much"
    ]):
        return boost("Stress Overload")

    # Low mood
    if any(w in text for w in [
        "sad", "low", "empty", "tired", "no motivation", "hopeless"
    ]):
        return boost("Low Mood")

    # Emotional instability
    if any(w in text for w in [
        "mood swing", "unstable", "confused", "mixed feelings"
    ]):
        return boost("Emotional Instability")

    return top3


# A function for final prediction 
# Primary flow:
#      1. LSTM predicts top-3 diseases
#      2. If best confidence >= 0.65 → trust LSTM
#      3. If confidence < 0.65 → send top-3 to Groq LLM for accurate selection
#    Returns: (final_disease: str, confidence: float, top3: list)

def final_prediction(text: str):

    top3 = predict_top3(text)
    top3 = filter_predictions(top3, text)

    best = top3[0]["label"]
    confidence = top3[0]["confidence"]

    # cap confidence (avoid overconfidence)
    confidence = min(confidence, 0.90)

    if confidence >= 0.80:
        final = best
    else:
        try:
            final = verify_with_llm(text, top3)
            if final not in [d["label"] for d in top3]:
                final = best
        except:
            final = best

    return final, confidence, top3