import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string


# Load model
model = None
tokenizer = None
le_sent = None
le_side = None
le_sev = None


def get_model():
    global model

    if model is None:
        model = load_model(
            "model/drug_review/drug_model.h5",
            compile=False
        )

    return model


def load_resources():
    global tokenizer, le_sent, le_side, le_sev

    if tokenizer is None:
        with open("model/drug_review/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    if le_sent is None:
        with open("model/drug_review/le_sent.pkl", "rb") as f:
            le_sent = pickle.load(f)

    if le_side is None:
        with open("model/drug_review/le_side.pkl", "rb") as f:
            le_side = pickle.load(f)

    if le_sev is None:
        with open("model/drug_review/le_sev.pkl", "rb") as f:
            le_sev = pickle.load(f)

MAX_LEN = 120

# Clean the text using NLP

def clean_review(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_text = clean_review


# Reject inputs that are too short or contain no medical keywords.
def is_valid_review(text: str) -> bool:
    if len(text.split()) < 3:
        return False

    keywords = [
    "pain", "nausea", "dizziness", "fatigue", "vomiting",
    "headache", "anxiety", "sleep", "side effect",
    "worked", "helped", "worse", "better", "reaction"
]

    return any(k in text.lower() for k in keywords)



# LSTM prediction - Return top-3 LSTM predictions with normalized confidence.

def predict_review(review: str) -> dict:
    load_resources()
    review = clean_review(review)

    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    current_model = get_model()
    sent_pred, side_pred, sev_pred = current_model.predict(pad, verbose=0)

    sentiment = le_sent.inverse_transform([np.argmax(sent_pred)])[0]
    side = le_side.inverse_transform([np.argmax(side_pred)])[0]
    severity = le_sev.inverse_transform([np.argmax(sev_pred)])[0]

    confidence = float(
    (np.max(sent_pred) + np.max(side_pred) + np.max(sev_pred)) / 3
)

    return {
        "sentiment": sentiment,
        "side_effects": side,
        "severity": severity,
        "confidence": confidence
    }


def apply_rules(pred, text):
    text = text.lower()

    # Strong no-side-effect rule
    if "no side effect" in text or "no side effects" in text:
        pred["side_effects"] = "No"
        pred["severity"] = "None"

    # Severe detection
    if any(w in text for w in ["severe", "extreme", "hospital", "er", "stop taking", "had to stop"]):
        pred["severity"] = "Severe"
        pred["side_effects"] = "Yes"

    # Moderate detection
    if any(w in text for w in ["persistent", "constant", "bad"]):
        pred["severity"] = "Moderate"

    # Mild detection
    if any(w in text for w in ["slight", "mild", "little"]):
        pred["severity"] = "Mild"

    # Contradiction handling
    if "but" in text and pred["sentiment"] == "Positive":
        pred["sentiment"] = "Neutral"

    return pred



#  Groq LLM verification 
# Use Groq LLaMA-3 to choose the most accurate diagnosis from LSTM top-3.Returns the disease name as a plain string.
def verify_with_llm(review: str, pred: dict):
    from utils.groq_client import generate_with_groq

    prompt = f"""
User review:
"{review}"

Model prediction:
Sentiment: {pred['sentiment']}
Side Effects: {pred['side_effects']}
Severity: {pred['severity']}

Task:
Correct the prediction if needed.

Rules:
- Keep output structured
- Do not give medical advice
- Only return corrected labels

Output format:
Sentiment: ...
Side Effects: ...
Severity: ...
"""

    return generate_with_groq(prompt)


def parse_llm_output(text):
    result = {}
    try:
        lines = text.split("\n")
        for line in lines:
            if "Sentiment:" in line:
                result["sentiment"] = line.split(":")[1].strip()
            elif "Side Effects:" in line:
                result["side_effects"] = line.split(":")[1].strip()
            elif "Severity:" in line:
                result["severity"] = line.split(":")[1].strip()
    except:
        pass
    return result



#  AI explanation

def explain_review(review, pred):
    from utils.groq_client import generate_with_groq

    prompt = f"""
User review:
"{review}"

Analysis:
Sentiment: {pred['sentiment']}
Side Effects: {pred['side_effects']}
Severity: {pred['severity']}

Generate a structured explanation in CLEAN HTML.

STRICT FORMAT:

<h4>Analysis</h4>
<p>Explain why this sentiment was predicted.</p>

<h4>Side Effects</h4>
<p>Explain detected side effects clearly.</p>

<h4>Severity Interpretation</h4>
<p>Explain how severe the effects are.</p>

<h4>Disclaimer</h4>
<p>This is not medical advice.</p>

RULES:
- ONLY return HTML
- NO markdown (**)
- NO extra text
"""

    return generate_with_groq(prompt)





# A function for final prediction 
# Primary flow:
#      1. LSTM predicts top-3 diseases
#      2. If best confidence >= 0.65 → trust LSTM
#      3. If confidence < 0.65 → send top-3 to Groq LLM for accurate selection
#    Returns: (final_disease: str, confidence: float, top3: list)

def final_review_analysis(review: str):
    
    pred = predict_review(review)
    pred = apply_rules(pred, review)
    if pred["confidence"] > 0.9:
        pred["confidence"] = 0.85
    if pred["confidence"] < 0.7 or pred["severity"] == "Severe":
        try:
            llm_result = verify_with_llm(review, pred)
            parsed = parse_llm_output(llm_result)
            VALID_SENT = ["Positive", "Neutral", "Negative"]
            VALID_SIDE = ["Yes", "No"]
            VALID_SEV  = ["None", "Mild", "Moderate", "Severe"]

            if parsed:
                if parsed.get("sentiment") in VALID_SENT:
                    pred["sentiment"] = parsed["sentiment"]

                if parsed.get("side_effects") in VALID_SIDE:
                    pred["side_effects"] = parsed["side_effects"]

                if parsed.get("severity") in VALID_SEV:
                    pred["severity"] = parsed["severity"]
        except:
            pass

    try:
        explanation = explain_review(review, pred)
    except:
        explanation = "This is an AI-based analysis for awareness only."

    pred["explanation"] = explanation

    return pred