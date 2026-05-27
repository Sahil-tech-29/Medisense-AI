import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Load saved model 
# model = load_model("model/symptom_model1.h5", compile=False) - change
model = None
tokenizer = None
le = None

def get_model():
    global model
    if model is None:
        model = load_model("model/symptom_model1.h5", compile=False)
    return model

def load_resources():
    global tokenizer, le

    if tokenizer is None:
        with open("model/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

    if le is None:
        with open("model/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

# Keeping the max length same as in ipynb file 
MAX_LEN = 120

# Disease descriptions 
DISEASE_DESCRIPTIONS = {
    "Fungal infection":      "A fungal infection caused by various fungi that can affect skin, nails, or internal organs.",
    "Allergy":               "An immune response to substances (allergens) that are generally harmless.",
    "GERD":                  "Gastroesophageal reflux disease — chronic acid reflux damaging the esophagus.",
    "Chronic cholestasis":   "Reduced bile flow from the liver, leading to jaundice and itching.",
    "Drug Reaction":         "Adverse response to a medication, ranging from rash to organ damage.",
    "Peptic ulcer disease":  "Sores in the lining of stomach or small intestine caused by acid.",
    "AIDS":                  "Advanced HIV infection that severely damages the immune system.",
    "Diabetes":              "Metabolic disorder causing high blood sugar due to insulin issues.",
    "Gastroenteritis":       "Inflammation of the stomach and intestines, usually from infection.",
    "Bronchial Asthma":      "Chronic respiratory condition causing airway inflammation and narrowing.",
    "Hypertension":          "Persistently elevated blood pressure that strains heart and arteries.",
    "Migraine":              "Recurring severe headaches often with nausea and light sensitivity.",
    "Cervical spondylosis":  "Age-related wear of spinal disks in the neck causing pain and stiffness.",
    "Paralysis (brain hemorrhage)": "Loss of muscle function due to bleeding in the brain.",
    "Jaundice":              "Yellowing of skin and eyes from excess bilirubin in the blood.",
    "Malaria":               "Mosquito-borne parasitic infection causing cyclical fever and chills.",
    "Chicken pox":           "Highly contagious viral illness causing itchy blister-like rash.",
    "Dengue":                "Mosquito-borne viral fever with severe joint and muscle pain.",
    "Typhoid":               "Bacterial infection from contaminated food/water causing prolonged fever.",
    "Hepatitis A":           "Viral liver infection spread through contaminated food or water.",
    "Hepatitis B":           "Viral liver infection spread through blood or bodily fluids.",
    "Hepatitis C":           "Viral liver infection primarily spread through blood contact.",
    "Hepatitis D":           "Liver infection that only occurs with Hepatitis B co-infection.",
    "Hepatitis E":           "Waterborne liver infection common in areas with poor sanitation.",
    "Alcoholic hepatitis":   "Liver inflammation caused by excessive alcohol consumption.",
    "Tuberculosis":          "Bacterial lung infection spread through airborne droplets.",
    "Common Cold":           "Viral upper respiratory tract infection causing runny nose and sneezing.",
    "Pneumonia":             "Lung infection causing air sacs to fill with fluid or pus.",
    "Dimorphic hemmorhoids(piles)": "Swollen veins in the rectum or anus causing discomfort or bleeding.",
    "Heart attack":          "Blocked blood supply to heart muscle causing tissue death.",
    "Varicose veins":        "Swollen, twisted veins usually appearing in legs.",
    "Hypothyroidism":        "Underactive thyroid producing insufficient hormones.",
    "Hyperthyroidism":       "Overactive thyroid producing excess hormones.",
    "Hypoglycemia":          "Abnormally low blood sugar causing weakness and confusion.",
    "Osteoarthritis":        "Degenerative joint disease causing cartilage breakdown.",
    "Arthritis":             "Inflammation of joints causing pain and stiffness.",
    "(vertigo) Paroxysmal Positional Vertigo": "Sudden brief episodes of dizziness from inner ear crystals.",
    "Acne":                  "Skin condition causing pimples from clogged hair follicles.",
    "Urinary tract infection": "Bacterial infection in the urinary system.",
    "Psoriasis":             "Autoimmune skin condition causing rapid skin cell buildup.",
    "Impetigo":              "Highly contagious bacterial skin infection common in children.",
}

DISEASE_PRECAUTIONS = {
    "Fungal infection":      ["Keep skin dry and clean", "Use antifungal powder", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":               ["Identify and avoid triggers", "Carry antihistamines", "Wear a medical alert bracelet", "Consult an allergist"],
    "GERD":                  ["Avoid trigger foods", "Eat smaller meals", "Don't lie down after eating", "Elevate head while sleeping"],
    "Diabetes":              ["Monitor blood sugar regularly", "Follow prescribed medication", "Maintain healthy diet", "Exercise daily"],
    "Malaria":               ["Use mosquito nets", "Take antimalarial medication", "Wear long-sleeved clothing", "Use insect repellent"],
    "Dengue":                ["Eliminate mosquito breeding sites", "Use repellents", "Stay hydrated", "Seek immediate medical care"],
    "Typhoid":               ["Drink only boiled/purified water", "Wash hands before eating", "Get vaccinated", "Avoid street food"],
    "Tuberculosis":          ["Complete full antibiotic course", "Cover mouth when coughing", "Ensure good ventilation", "Regular check-ups"],
    "Pneumonia":             ["Complete antibiotic course", "Rest adequately", "Stay hydrated", "Get follow-up chest X-ray"],
    "Hypertension":          ["Monitor BP regularly", "Reduce salt intake", "Exercise regularly", "Limit alcohol and avoid smoking"],
    "Migraine":              ["Identify and avoid triggers", "Rest in dark quiet room", "Stay hydrated", "Take medication at first sign"],
    "Common Cold":           ["Rest and stay hydrated", "Wash hands frequently", "Avoid close contact with others", "Use saline nasal rinse"],
    "Chicken pox":           ["Avoid scratching blisters", "Keep skin clean", "Isolate from unvaccinated people", "Take prescribed antivirals"],
    "Gastroenteritis":       ["Stay hydrated with ORS", "Eat bland foods", "Wash hands thoroughly", "Avoid dairy temporarily"],
    "Heart attack":          ["Call emergency services immediately", "Chew aspirin if not allergic", "Rest and stay calm", "Do not drive yourself"],
    "Hepatitis A":           ["Get vaccinated", "Wash hands after toilet use", "Avoid contaminated food/water", "Rest and stay hydrated"],
    "Jaundice":              ["Consult doctor immediately", "Avoid alcohol completely", "Rest adequately", "Eat light nutritious food"],
    "Urinary tract infection": ["Drink plenty of water", "Complete antibiotic course", "Urinate frequently", "Maintain genital hygiene"],
    "Acne":                  ["Cleanse face gently twice daily", "Avoid touching face", "Use non-comedogenic products", "Stay hydrated"],
    "Psoriasis":             ["Moisturize skin regularly", "Avoid triggers like stress", "Use prescribed topical treatments", "Limit sun exposure carefully"],
}

DEFAULT_PRECAUTIONS = ["Consult a doctor immediately", "Rest and stay hydrated", "Avoid self-medication", "Monitor symptoms closely"]


# Clean the text using NLP

def clean_symptom_input(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Alias used in routes
clean_text = clean_symptom_input

# Reject inputs that are too short or contain no medical keywords.
def is_valid_symptom_input(text: str) -> bool:
    if len(text.split()) < 2:
        return False
    medical_keywords = [
        "fever", "pain", "cough", "headache", "fatigue", "nausea", "vomit",
        "rash", "itch", "cold", "chills", "sweat", "dizzy", "diarrhea",
        "breath", "chest", "throat", "stomach", "skin", "joint", "muscle",
        "bleed", "swell", "weak", "loss", "burning", "discharge", "yellow",
        "sneeze", "runny", "congestion", "blur", "urine", "bowel", "back",
        "neck", "abdomen", "heart", "blood", "weight", "appetite", "sleep",
        "anxiety", "numbness", "tingling", "tremor", "sore", "dry", "excess",
    ]
    return any(kw in text for kw in medical_keywords)


# Metadata helpers 
def get_description(disease: str) -> str:
    return DISEASE_DESCRIPTIONS.get(disease, f"{disease} — a medical condition requiring professional evaluation.")


def get_precautions(disease: str) -> list:
    return DISEASE_PRECAUTIONS.get(disease, DEFAULT_PRECAUTIONS)

# Keyword-based severity estimate.
def calculate_severity(symptoms: str) -> str:
    
    HIGH = ["chest pain", "difficulty breathing", "heart attack", "paralysis",
            "hemorrhage", "unconscious", "blood", "severe pain", "stroke"]
    MEDIUM = ["fever", "vomiting", "diarrhea", "rash", "swelling",
              "headache", "dizziness", "jaundice", "weakness"]
    text = symptoms.lower()
    if any(kw in text for kw in HIGH):
        return "High — Seek emergency care immediately"
    if any(kw in text for kw in MEDIUM):
        return "Moderate — Consult a doctor soon"
    return "Low — Monitor symptoms; visit a clinic if worsening"


# LSTM prediction - Return top-3 LSTM predictions with normalized confidence.

def predict_top3(symptoms: str) -> list:
    load_resources()
    symptoms = clean_symptom_input(symptoms)

    seq = tokenizer.texts_to_sequences([symptoms])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # Taking help of probability
    # probs = model.predict(pad, verbose=0)[0]--change

    current_model = get_model()
    probs = current_model.predict(pad, verbose=0)[0]

    probs = probs.flatten()
    def calibrate_probs(probs, temperature=1.5):
        logits = np.log(probs + 1e-9)
        scaled = np.exp(logits / temperature)
        return scaled / np.sum(scaled)

    probs = calibrate_probs(probs)

    top3_idx = np.argsort(probs)[-3:][::-1]

    results = [
        {
            "disease": le.inverse_transform([i])[0],
            "confidence": float(probs[i])
        }
        for i in top3_idx
    ]
    for r in results:
        r["confidence"] = min(r["confidence"], 0.85)

    total = sum(r["confidence"] for r in results)
    if total > 0:
        for r in results:
            r["confidence"] = r["confidence"] / total

    return results


#  Groq LLM verification 
# Use Groq LLaMA-3 to choose the most accurate diagnosis from LSTM top-3.Returns the disease name as a plain string.
def verify_with_llm(symptoms: str, top3: list) -> str:
    from utils.groq_client import generate_with_groq

    options_block = "\n".join(
        f"  {i+1}. {d['disease']} (LSTM confidence: {d['confidence']:.1%})"
        for i, d in enumerate(top3)
    )

    prompt = f"""You are a senior clinical decision-support AI. A patient has described the following symptoms:

Patient symptoms: "{symptoms}"

An LSTM model returned these possible diagnoses (low confidence — needs verification):
{options_block}

Your task:
1. Carefully analyze the symptom description.
2. Select the SINGLE most medically accurate diagnosis from the list above.
3. If none of the options fit the symptoms well, you may suggest a better disease name.
4. Be conservative — in medicine, accuracy is paramount.

Rules:
- Do NOT guess randomly.
- Do NOT explain or add any commentary.
- Return ONLY the disease name, exactly as written above (or your corrected name).
- One line, no punctuation at the end.
"""
    return generate_with_groq(prompt).strip()


#  AI explanation

# Generate an HTML-formatted clinical explanation of why these symptoms match the predicted disease, using Groq LLaMA-3.
def ai_explain_disease(symptoms: str, disease: str) -> str:
    from utils.groq_client import generate_with_groq

    prompt = f"""You are a medical education assistant. A patient described these symptoms:
"{symptoms}"

The predicted diagnosis is: {disease}

Write a concise, patient-friendly explanation in HTML (no <html>/<body> tags). Include:
1. Why these symptoms point to {disease}
2. What {disease} is (1-2 sentences)
3. When to seek immediate medical attention
4. A clear disclaimer that this is AI-generated and not a substitute for professional diagnosis

Use these HTML tags only: <h4>, <p>, <ul>, <li>, <strong>, <em>.
Keep it under 200 words. Be medically accurate and empathetic.
"""
    result = generate_with_groq(prompt).strip()
    # Fallback if Groq fails or returns empty
    if not result:
        desc = get_description(disease)
        return f"<p><strong>{disease}:</strong> {desc}</p><p><em>Please consult a qualified doctor for an accurate diagnosis.</em></p>"
    return result


#  Main prediction pipeline 
SYMPTOM_MAP = {
    "Common Cold": ["cough", "runny nose", "sore throat"],
    "Tonsillitis": ["sore throat", "fever"],
    "Dengue": ["fever", "headache", "pain behind eyes"],
    "UTI": ["burning", "urination"],
    "Pneumonia": ["cough", "fever", "chest pain"],
}

def symptom_match_score(disease, symptoms):
    disease_symptoms = SYMPTOM_MAP.get(disease, [])
    match = sum(1 for s in disease_symptoms if s in symptoms.lower())
    return match / (len(disease_symptoms) + 1e-5)

# if having following keywords in the text provided by user 
def filter_predictions(top3, symptoms):
    symptoms = symptoms.lower()

    # Skin
    if any(w in symptoms for w in ["rash", "itch", "skin", "blister"]):
        allowed = ["Allergy", "Skin Rash", "Fungal Infection", "Eczema", "Hives", "Chickenpox"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Gastro 
    if any(w in symptoms for w in ["vomit", "diarrhea", "nausea"]):
        allowed = ["Gastroenteritis", "Food Poisoning", "Diarrhea", "IBS"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Urinary
    if any(w in symptoms for w in ["urine", "pee", "burning"]):
        allowed = ["UTI", "Mild Kidney Infection", "Kidney Stones"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Joint
    if any(w in symptoms for w in ["joint", "stiff", "swelling"]):
        allowed = ["Arthritis", "Muscle Strain", "Joint Inflammation"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Head
    if "head" in symptoms and not any(w in symptoms for w in ["fever", "vomit"]):
        allowed = ["Migraine", "Tension Headache"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Chest
    if "chest" in symptoms:
        allowed = ["Angina Awareness", "Pneumonia"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Back
    if "back" in symptoms:
        allowed = ["Back Pain", "Kidney Stones", "Mild Kidney Infection"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Abdomen
    if any(w in symptoms for w in ["abdomen", "stomach", "belly"]):
        allowed = ["Appendicitis", "Gastroenteritis", "Food Poisoning", "IBS"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    # Generic
    if any(w in symptoms for w in ["not feeling", "tired", "weak", "uncomfortable"]):
        allowed = ["Low Risk / General Condition", "Viral Fever", "Common Cold"]
        filtered = [d for d in top3 if d["disease"] in allowed]
        return filtered if filtered else top3

    #  Default similarity
    filtered = []
    for item in top3:
        score = symptom_match_score(item["disease"], symptoms)
        if score >= 0.3:
            filtered.append(item)

    return filtered if len(filtered) >= 3 else top3


# A function for final prediction 
# Primary flow:
#      1. LSTM predicts top-3 diseases
#      2. If best confidence >= 0.65 → trust LSTM
#      3. If confidence < 0.65 → send top-3 to Groq LLM for accurate selection
#    Returns: (final_disease: str, confidence: float, top3: list)

def final_prediction(symptoms: str) -> tuple:
    
    top3 = predict_top3(symptoms)
    top3 = filter_predictions(top3, symptoms)
    best_disease = top3[0]["disease"]
    confidence = top3[0]["confidence"]
    
    if confidence > 0.9:
        confidence = 0.85
    if confidence >= 0.85:
        # LSTM is confident — use it directly
        final = best_disease
    else:
        # Low confidence → use Groq to pick the most accurate option
        try:
            final = verify_with_llm(symptoms, top3)
            # Safety: if Groq returns something unrecognised, fall back to LSTM top-1
            if not final or len(final) < 3:
                final = best_disease
        except Exception:
            final = best_disease

    return final, confidence, top3