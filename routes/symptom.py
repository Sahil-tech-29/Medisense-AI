from flask import Blueprint, render_template, request, session, redirect, url_for
from services.symptom import (
    clean_symptom_input,
    ai_explain_disease,
    is_valid_symptom_input
)
from utils.auth import login_required
from utils.activity_logger import log_activity

symptom_bp = Blueprint("symptom", __name__)

@symptom_bp.route("/symptom", methods=["GET", "POST"])
@login_required
def symptom_page():

    #  POST 
    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()

        if not symptoms:
            return redirect(url_for("symptom.symptom_page"))

        cleaned = clean_symptom_input(symptoms)

        if not is_valid_symptom_input(cleaned):
            session["last_symptom_text"] = symptoms
            session["last_symptom_disease"] = "Input not understood"
            session["last_symptom_explanation"] = (
                "<p>Please enter meaningful symptoms like fever, pain, cough, or headache.</p>"
            )
            return redirect(url_for("symptom.symptom_page"))

        # AI CALLS ONLY HERE
        from services.symptom import final_prediction, get_description, get_precautions, calculate_severity

        disease, confidence, top3 = final_prediction(cleaned)

        desc = get_description(disease)
        precautions = get_precautions(disease)
        risk = calculate_severity(cleaned)


        explanation = ai_explain_disease(cleaned, disease)

        # Save to session
        session["last_symptom_text"] = symptoms 
        session["last_symptom_disease"] = str(disease)
        session["last_symptom_desc"] = str(desc)
        session["last_symptom_precautions"] = list(precautions)
        session["last_symptom_risk"] = str(risk)
        session["last_symptom_confidence"] = float(confidence)
        session["last_symptom_explanation"] = str(explanation)
        session["top3"] = top3 if top3 else []

        #  Save to DB ONCE
        log_activity(
            module="symptom",
            user_input=symptoms,
            ai_output=f"Disease: {disease}\n\n{explanation}"
        )

        return redirect(url_for("symptom.symptom_page"))

    # GET 
    return render_template(
    "symptom.html",
    symptoms=session.get("last_symptom_text", ""),
    disease=session.get("last_symptom_disease"),
    explanation=session.get("last_symptom_explanation"),
    desc=session.get("last_symptom_desc"),
    precautions=session.get("last_symptom_precautions"),
    risk=session.get("last_symptom_risk"),
    confidence=session.get("last_symptom_confidence"),
    top3=session.get("top3")
)