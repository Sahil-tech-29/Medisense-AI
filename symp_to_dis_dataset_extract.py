import random
import pandas as pd

random.seed(42)


DISEASE_SYMPTOMS = {
    "Common Cold": [
        "cough runny nose sneezing",
        "mild cold and cough",
        "sore throat sneezing",
        "runny nose mild fever",
        "cold cough"
    ],
    "Viral Fever": [
        "fever headache body pain",
        "high temperature fatigue",
        "fever with weakness",
        "fever chills body ache"
    ],
    "Flu": [
        "fever cough fatigue",
        "body pain sore throat",
        "flu like symptoms"
    ],
    "Allergy": [
        "sneezing itchy eyes",
        "runny nose allergy",
        "itchy throat sneezing"
    ],
    "Migraine": [
        "severe headache nausea",
        "one sided head pain",
        "headache sensitivity to light"
    ],
    "Food Poisoning": [
        "vomiting diarrhea stomach pain",
        "loose motion nausea",
        "abdominal pain vomiting"
    ],
    "Gastritis": [
        "stomach pain acidity",
        "burning sensation in stomach",
        "indigestion nausea"
    ],
    "Urinary Tract Infection": [
        "burning urination",
        "frequent urination pain",
        "lower abdominal pain during urination"
    ],
    "Skin Infection": [
        "skin redness swelling",
        "painful skin rash",
        "infected skin area"
    ],
    "Fungal Infection": [
        "itching skin rash",
        "red patches fungal infection",
        "skin itching discoloration"
    ],
    "Conjunctivitis": [
        "red eyes discharge",
        "itchy watery eyes",
        "eye redness pain"
    ],
    "Sinusitis": [
        "facial pain nasal congestion",
        "sinus headache pressure",
        "blocked nose headache"
    ],
    "Bronchitis": [
        "persistent cough mucus",
        "chest congestion cough",
        "breathing difficulty cough"
    ],
    "Asthma": [
        "shortness of breath wheezing",
        "chest tightness breathing problem",
        "asthma attack wheezing"
    ],
    "Hypertension": [
        "high blood pressure headache",
        "dizziness blurred vision",
        "pressure in head"
    ],
    "Diabetes": [
        "frequent urination thirst",
        "fatigue slow healing",
        "increased hunger tiredness"
    ],
    "Anemia": [
        "fatigue weakness dizziness",
        "pale skin tiredness",
        "shortness of breath fatigue"
    ],
    "Pneumonia": [
        "fever cough chest pain",
        "breathing difficulty fever",
        "productive cough weakness"
    ],
    "Typhoid": [
        "high fever abdominal pain",
        "weakness loss of appetite",
        "fever for many days"
    ],
    "Dengue": [
        "high fever joint pain",
        "body ache headache",
        "fever rash pain"
    ],
    "Malaria": [
        "fever chills sweating",
        "intermittent fever weakness",
        "shivering fever"
    ],
    "COVID-19": [
        "fever cough loss of taste",
        "breathing difficulty fatigue",
        "covid symptoms"
    ],
    "Back Pain": [
        "lower back pain stiffness",
        "back ache movement difficulty"
    ],
    "Arthritis": [
        "joint pain swelling",
        "morning stiffness joints"
    ],
    "Anxiety": [
        "restlessness racing thoughts",
        "anxiety panic symptoms"
    ],
    "Depression": [
        "persistent sadness fatigue",
        "loss of interest sleep problems"
    ],
    "Insomnia": [
        "difficulty sleeping",
        "frequent night waking"
    ]
}

# ==============================
# LONG DESCRIPTION HELPERS
# ==============================

DURATIONS = [
    "one day", "two days", "three days",
    "a week", "ten days"
]

SEVERITY = [
    "mild", "moderate",
    "manageable", "gradually worsening"
]

LONG_TEMPLATES = [
    "i have been experiencing {symptoms} for the past {duration}. "
    "the symptoms are {severity} and affecting my daily routine.",

    "for the last {duration}, i am dealing with {symptoms}. "
    "sometimes it feels {severity}, especially during the day.",

    "since {duration}, i noticed {symptoms}. "
    "there is no severe pain but the discomfort is {severity}.",

    "i am feeling {symptoms}. this has been happening for {duration} "
    "and it has started to worry me."
]

# ==============================
# TEXT GENERATOR
# ==============================

def generate_text(symptoms):
    r = random.random()

    # 40% short
    if r < 0.4:
        return f"{symptoms}"

    # 30% medium
    if r < 0.7:
        return f"i have {symptoms} for {random.choice(DURATIONS)}"

    # 30% long
    template = random.choice(LONG_TEMPLATES)
    return template.format(
        symptoms=symptoms,
        duration=random.choice(DURATIONS),
        severity=random.choice(SEVERITY)
    )

# ==============================
# DATASET GENERATION
# ==============================

ROWS_PER_DISEASE = 3500   # 🔥 change to 3500+ for ~100k rows

rows = []

for disease, symptom_list in DISEASE_SYMPTOMS.items():
    for _ in range(ROWS_PER_DISEASE):
        base_symptom = random.choice(symptom_list)
        text = generate_text(base_symptom)
        rows.append([disease, text])

df = pd.DataFrame(rows, columns=["label", "text"])
df = df.sample(frac=1).reset_index(drop=True)

print("Total rows:", len(df))
print(df.head())

df.to_csv("synthetic_symptom_disease_dataset.csv", index=False)
print("Saved → synthetic_symptom_disease_dataset.csv")
