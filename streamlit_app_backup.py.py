import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib 

from medical_report.pdf_extraction import extract_text_from_pdf
from medical_report.report_summarizer import summarize_medical_report

from preventive_health.risk_engine import calculate_health_risks
from preventive_health.prompt_utils import build_preventive_prompt

from groq import Groq

import string
import re



# page config


if "page" not in st.session_state:
    st.session_state.page = "home"

st.set_page_config(
    page_title="Disease Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Global CSS 
# CSS FOR CARDS


st.markdown("""
<style>

/* --- CARD AS BUTTON --- */
div.stButton > button {
    width: 100%;
    height: 130px;                 /* SAME HEIGHT */
    background: white;
    padding: 22px 24px;
    border-radius: 20px;
    border: none;
    box-shadow: 0 10px 26px rgba(0,0,0,0.08);
    text-align: left;
    transition: all 0.25s ease;

    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 6px;
}

/* Hover */
div.stButton > button:hover {
    transform: translateY(-6px);
    box-shadow: 0 22px 40px rgba(0,0,0,0.14);
    background: #f8fafc;
}

/* Focus */
div.stButton > button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.25);
}

/* TITLE (first line) */
div.stButton > button::first-line {
    font-size: 20px;
    font-weight: 800;
    color: #0F172A;
}

/* DESCRIPTION */
div.stButton > button span {
    font-size: 14px;
    color: #64748B;
}

/* Remove default spacing */
div.stButton > button p {
    margin: 0;
}

</style>
""", unsafe_allow_html=True)




if st.session_state.page == "home":

    st.markdown("<h1 style='text-align:center;'>🧠 Medisense AI</h1>", unsafe_allow_html=True)
    st.caption("AI-powered healthcare awareness platform")

    # ROW 1
col1, col2 = st.columns(2, gap="large")

with col1:
    if st.button(
        "🩺  𝗗𝗜𝗦𝗘𝗔𝗦𝗘 𝗣𝗥𝗘𝗗𝗜𝗖𝗧𝗜𝗢𝗡\n\n"
        "Predict possible conditions using symptoms"
    ):
        st.session_state.page = "symptom"
        st.rerun()


with col2:
    if st.button(
        "🧠  Mental Health Detection\n\n"
        "Detect emotional patterns using AI"
    ):
        st.session_state.page = "mental"
        st.rerun()


st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# ROW 2
col3, col4 = st.columns(2, gap="large")

with col3:
    if st.button(
        "📄  Medical Report Summary\n\n"
        "Upload PDF and get simplified explanation"
    ):
        st.session_state.page = "pdf"
        st.rerun()

with col4:
    if st.button(
        "💊  Drug Side-Effect Risk\n\n"
        "Analyze user drug experiences"
    ):
        st.session_state.page = "drug"
        st.rerun()


st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# ROW 3 (CENTERED)
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    if st.button(
        "🌿  Preventive Health Awareness\n\n"
        "Lifestyle-based health risk analysis"
    ):
        st.session_state.page = "preventive"
        st.rerun()








# load model & tools

@st.cache_resource
def load_symptom_model():
    model = tf.keras.models.load_model(
        "model/symptom_model1.h5",
        compile=False
    )   

    with open("model/symptom/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("model/symptom/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder


@st.cache_resource
def load_mental_health_model():
    model = tf.keras.models.load_model(
        "model/mental_health/mental_health_lstm_model.h5"
    )
    tokenizer = joblib.load(
        "model/mental_health/mh_tokenizer.pkl"
    )
    label_encoder = joblib.load(
        "model/mental_health/mh_label_encoder.pkl"
    )
    return model, tokenizer, label_encoder


@st.cache_resource
def load_drug_review_model():
    model = tf.keras.models.load_model("model/drug_review/drug_side_effect_lstm.h5")
    with open("model/drug_review/drug_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer








# Helper functions 

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



SEVERE_TERMS = [
    "death", "fatal", "bleeding", "seizure",
    "stroke", "heart attack", "organ failure",
    "coma", "paralysis", "anaphylaxis"
]

COMMON_SIDE_EFFECTS = [
    "nausea", "vomiting", "dizziness", "headache",
    "fatigue", "rash", "diarrhea", "anxiety",
    "insomnia", "dry mouth", "constipation"
]

def extract_side_effects(text):
    text = text.lower()
    found = [effect for effect in COMMON_SIDE_EFFECTS if effect in text]
    return list(set(found))


def severity_score(text):
    text = text.lower()
    return sum(term in text for term in SEVERE_TERMS)


def confidence_band(conf):
    if conf >= 85:
        return "High Confidence"
    elif conf >= 65:
        return "Moderate Confidence"
    else:
        return "Low Confidence"






# GROQ IMPLEMENTATION 

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
def generate_with_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a healthcare awareness assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=700
    )
    return response.choices[0].message.content.strip()






# Groq implementation in Symptom - based prediction 

def ai_explain_disease(symptoms, disease):
    prompt = f"""
    A system has identified the condition: "{disease}"

    User symptoms:
    {symptoms}

    Explain the condition in detail with the following sections:

    1. What this condition generally means
    2. Why these symptoms match it
    3. Whether it is usually mild or serious
    4. General self-care and precautions (non-medical)
    5. When to consult a doctor

    Rules:
    - Do NOT change the disease
    - Do NOT diagnose definitively
    - Do NOT prescribe medicine
    - Educational tone only
    """
    return generate_with_groq(prompt)






# Groq AI  in mental health prediction 

def generate_mental_health_explanation(text, label, confidence, risk):
    prompt = f"""
    A mental health AI system analyzed the user's text and detected patterns
    related to "{label}" with a confidence of {confidence*100:.2f}%.
    Risk level: {risk}.

    User text:
    {text}

    Provide a supportive, empathetic explanation with the following sections:

    1. What this mental state generally means (simple, non-clinical language)
    2. Why such feelings or thoughts can occur (no assumptions)
    3. Gentle self-care or coping suggestions (non-medical)
    4. When it may be helpful to talk to someone (friend, family, or professional)
    5. Reassurance that seeking help is okay

    Important rules:
    - DO NOT diagnose
    - DO NOT label the user with a disorder
    - DO NOT prescribe medication
    - Keep the tone calm, empathetic, and human
    - Use bullet points where appropriate
    - Avoid alarming language
    """

    return generate_with_groq(prompt)






# Groq AI in drug review 

def generate_drug_risk_explanation(review, risk_label, confidence):
    prompt = f"""
    You are assisting in a healthcare awareness system.

    A machine learning model analyzed the following user-written drug review and classified
    the side-effect risk as "{risk_label}" with a confidence of {confidence:.2f}%.

    User review:
    \"\"\"{review}\"\"\"

    TASK:
    Generate a clear, review-specific explanation strictly based on the content above.

    Follow this exact structure:

    1. Side effects mentioned in the review  
    - List only the side effects explicitly stated or strongly implied in the text  
    - If none are clear, say "No specific side effects clearly mentioned"

    2. What this risk level means in this context  
    - Explain what "{risk_label}" means **for this review only**
    - Do NOT generalize beyond the given text

    3. Why these effects may be concerning or manageable  
    - If risk is "Risky": explain why the mentioned effects could be concerning  
    - If risk is "Likely Safe": explain why the effects appear mild or tolerable  

    4. Safety awareness guidance (non-medical)  
    - Focus on awareness and observation, not treatment  

    5. When to consider professional help  
    - Use cautious, non-alarming language  

    IMPORTANT RULES:
    - Do NOT diagnose
    - Do NOT recommend medicines, treatments, or dosages
    - Do NOT introduce side effects not present in the review
    - Do NOT give generic health advice
    - Do NOT repeat disclaimers excessively
    - Keep the explanation grounded in the user's review
    - Use short bullet points, not long paragraphs
    - Maintain a calm, educational tone
    """

    return generate_with_groq(prompt)







# UI for symptom based 

if st.button("⬅ Back to Home"):
    st.session_state.page = "home"
    st.rerun()


if st.session_state.page == "symptom":


    st.title("🩺 Symptom-Based Disease Prediction")
    st.write("Enter your symptoms in plain English:")

    user_input = st.text_area(
        "Symptoms",
        placeholder="e.g. chest pain, sweating, shortness of breath"
    )

    if st.button("Predict Disease"):
        if user_input.strip() == "":
            st.warning("Please enter symptoms.")
        else:
            if len(user_input.split()) <= 4:
                user_input = f"I have {user_input.replace(',', ' and ')} for a short duration."

            with st.spinner("Analyzing symptoms..."):
                predicted_disease = ai_predict_disease(user_input)

            
            st.success(f"🧾 Predicted Disease: **{predicted_disease}**")

            
            st.write("🔢 Confidence: **High**")

            with st.spinner("Generating explanation..."):
                explanation = ai_explain_disease(user_input, predicted_disease)

            st.markdown(
                f"""
                <div style="
                    background-color:#E8F5E9;
                    padding:18px;
                    border-radius:12px;
                    color:#1B5E20;
                    font-size:16px;
                    line-height:1.6;
                ">
                    <b>🧠 AI Explanation</b><br><br>
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                ⚠️ **Disclaimer:**  
                This system provides health-related awareness using AI
                and does not replace professional medical advice.
                """
            )







# ui for mental health 
if st.session_state.page == "mental":

    mh_model, mh_tokenizer, mh_label_encoder = load_mental_health_model()
    MAX_LEN = 120

    st.title("🧠 Mental Health Detection")
    st.write("Describe how you have been feeling recently:")

    user_text = st.text_area(
        "Your thoughts / feelings",
        placeholder="e.g. I feel anxious and exhausted lately..."
    )

    if st.button("Analyze Mental Health"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            seq = mh_tokenizer.texts_to_sequences([user_text.lower()])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

            pred = mh_model.predict(pad)
            class_id = pred.argmax()
            confidence = pred.max()

            label = mh_label_encoder.inverse_transform([class_id])[0]

            st.success(f"🧾 Detected Pattern: **{label}**")
            st.write(f"🔢 Confidence: **{confidence*100:.2f}%**")

            risk = "Low" if confidence < 0.4 else "Moderate" if confidence < 0.7 else "High"
            st.info(f"⚠️ Risk Level: **{risk}**")

            with st.spinner("Generating AI explanation..."):
                explanation = generate_mental_health_explanation(
                    user_text,
                    label,
                    confidence,
                    risk
                )


            st.markdown(
                f"""
                <div style="
                    background-color:#E8F5E9;
                    padding:18px;
                    border-radius:12px;
                    color:#1B5E20;
                    font-size:16px;
                    line-height:1.6;
                ">
                    <b>🧠 AI Explanation</b><br><br>
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown(
                "⚠️ **Disclaimer:** This is not a medical diagnosis. Please seek professional help if needed."
            )








# UI for pdf extraction

if st.session_state.page == "pdf":


    st.title("📄 Medical Report Summarization")
    st.info(
        "Upload a medical PDF report to receive a simplified explanation. "
        "This feature is for educational purposes only."
    )

    uploaded_pdf = st.file_uploader(
        "Upload Medical Report (PDF only)",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        with st.spinner("Extracting text from PDF..."):
            report_text = extract_text_from_pdf(uploaded_pdf)

        from medical_report.pdf_extraction import is_readable_text

        if not is_readable_text(report_text):
            st.error(
                "❌ This PDF appears to be scanned or image-based.\n\n"
                "Text extraction is not possible without OCR.\n"
                "Please upload a digitally generated PDF report."
            )

        else:
            st.success("PDF text extracted successfully!")

            with st.expander("🔍 View Extracted Text"):
                st.text(report_text[:6000])

            if st.button("🧠 Generate AI Summary"):
                with st.spinner("Generating summary using AI..."):
                    summary = summarize_medical_report(report_text)

                st.subheader("📘 Simplified Report Summary")
                st.write(summary)

                st.warning(
                    "⚠️ Disclaimer: This summary is generated for awareness only "
                    "and does not replace professional medical advice."
                )









# UI for Drug Side-Effect Risk Detection

if st.session_state.page == "drug":

    model, tokenizer = load_drug_review_model()
    MAX_LEN = 120

    st.title("🧪 Drug Side-Effect Risk Detection")
    st.write(
        "Enter a drug-related experience or review. "
        "The system analyzes user-reported experiences for awareness purposes."
    )

    st.info(
        "Severe reactions may include bleeding, seizures, organ damage, or emergency symptoms. "
        "Mild effects such as headache or nausea are usually low risk."
    )

    user_review = st.text_area(
        "Drug Review",
        placeholder="e.g. I experienced severe nausea and dizziness after taking this medicine..."
    )

    if st.button("Analyze Side-Effect Risk"):
        if user_review.strip() == "":
            st.warning("Please enter a drug review.")
        else:
            
            # Model Prediction
            
            seq = tokenizer.texts_to_sequences([user_review.lower()])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

            raw_prob = model.predict(pad)[0][0]

            # Uncertainty Handling
            
            if 0.45 <= raw_prob <= 0.55:
                st.warning(
                    "⚠️ The model is uncertain. The review contains mixed or unclear signals."
                )

            # Class-aware Confidence

            if raw_prob >= 0.5:
                risk_label = "Risky"
                confidence = raw_prob * 100
                st.error(f"⚠️ Side-Effect Risk: **{risk_label}**")
            else:
                risk_label = "Likely Safe"
                confidence = (1 - raw_prob) * 100
                st.success(f"✅ Side-Effect Risk: **{risk_label}**")

            
            # Confidence Bar + Band

            st.write(f"📊 Prediction Certainity: **{confidence:.2f}%**")
            band = confidence_band(confidence)
            st.write(f"🔍 Confidence Level: **{band}**")

            st.caption(
                "Confidence indicates how certain the system is about this assessment, "
                "not how severe the side effects are."
            )

           
            # Detected Side Effects
           
            detected_effects = extract_side_effects(user_review)

            if detected_effects:
                st.subheader("🧾 Detected Side Effects")
                for eff in detected_effects:
                    st.write(f"- {eff.capitalize()}")
            else:
                st.subheader("🧾 Detected Side Effects")
                st.write("No common side effects explicitly detected.")

           
            # Severity Indicator
           
            sev_score = severity_score(user_review)

            if sev_score >= 2:
                st.error("🚨 Multiple severe side-effect indicators detected")
            elif sev_score == 1:
                st.warning("⚠️ One severe side-effect indicator detected")
            else:
                st.info("ℹ️ No explicit severe side-effect indicators found")

        
            # Hybrid Decision Wording
         
            st.subheader("🧠 Final Assessment")

            if risk_label == "Risky" and sev_score >= 1:
                st.error(
                    "Overall assessment indicates elevated risk due to both model prediction "
                    "and presence of severe indicators."
                )
            elif risk_label == "Likely Safe" and sev_score >= 2:
                st.warning(
                    "Model predicts low risk, but severe indicators were detected. "
                    "Caution is advised."
                )
            else:
                st.success(
                    "Model prediction and severity indicators are consistent."
                )

            # GenAI Explanation
   
            with st.spinner("Generating AI explanation..."):
                explanation = generate_drug_risk_explanation(
                    user_review,
                    risk_label,
                    confidence
                )

            st.markdown(
                f"""
                <div style="
                    background-color:#FFF3E0;
                    padding:18px;
                    border-radius:12px;
                    color:#E65100;
                    font-size:16px;
                    line-height:1.6;
                ">
                    <b>🧠 AI Explanation</b><br><br>
                    {explanation}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                ⚠️ **Disclaimer:**  
                This analysis is based on user-reported experiences and is intended
                for awareness only. It does not replace professional medical advice.
                """
            )








# UI for preventive health risk awareness

if st.session_state.page == "preventive":


    st.title("🌿 Preventive Health Risk Awareness")
    st.info(
        "This module provides preventive health awareness based on lifestyle factors. "
        "It does NOT diagnose diseases or predict medical conditions."
    )

    with st.form("preventive_health_form"):
        age_group = st.selectbox(
            "Age Group",
            [
                "0 - 12",
                "13 - 18",
                "18 - 25",
                "26 - 40",
                "41 - 60",
                "60+"
            ]
        )

        activity = st.selectbox(
            "Physical Activity Level",
            ["High", "Moderate", "Low"]
        )
        diet = st.selectbox(
            "Diet Pattern",
            ["Balanced", "Mixed", "Junk / Irregular"]
        )
        sleep = st.slider(
            "Average Sleep Duration (hours)",
            min_value=1,
            max_value=15,
            value=7
        )

        stress = st.slider(
            "Stress Level (1 = Low, 5 = High)",
            1, 5, 3
        )
        smoking = st.selectbox(
            "Smoking Habit",
            ["No", "Occasional", "Regular"]
        )
        alcohol = st.selectbox(
            "Alcohol Consumption",
            ["No", "Occasional", "Frequent"]
        )

        submit = st.form_submit_button("Assess Preventive Health")
        age_mapping = {
        "0 - 12": 8,
        "13 - 18": 16,
        "18 - 25": 22,
        "26 - 40": 33,
        "41 - 60": 50,
        "60+": 65
    }

    age = age_mapping[age_group]

    if submit:
        user_inputs = {
            "age": age,
            "activity": activity,
            "diet": diet,
            "sleep": sleep,
            "stress": stress,
            "smoking": smoking,
            "alcohol": alcohol
        }

        risk_results = calculate_health_risks(user_inputs)

        st.subheader("🧠 Risk Awareness Summary")

        for risk, data in risk_results.items():

            color = (
                "#2E7D32" if data["level"] == "Low"
                else "#F9A825" if data["level"] == "Moderate"
                else "#C62828"
            )

            st.markdown(
                f"""
                <div style="
                    border-left: 6px solid {color};
                    background-color: #F8F9FA;
                    padding: 12px 16px;
                    border-radius: 8px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size:16px; font-weight:600;">
                        {risk}
                    </div>
                    <div style="font-size:14px; color:{color};">
                        Risk Level: <b>{data['level']}</b> &nbsp; | &nbsp;
                        Confidence: {int(data['confidence'] * 100)}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with st.spinner("Generating preventive guidance..."):
            prompt = build_preventive_prompt(risk_results)

            response_text = generate_with_groq(prompt)

        st.markdown(
            f"""
            <div style="
                background-color:#E3F2FD;
                padding:18px;
                border-radius:12px;
                color:#0D47A1;
                font-size:16px;
                line-height:1.6;
            ">
                <b>🧠 AI Preventive Guidance</b><br><br>
                {response_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            ⚠️ **Disclaimer:**  
            This assessment is for preventive awareness only and does not replace
            professional medical consultation.
            """
        )
