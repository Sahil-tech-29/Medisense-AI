from flask import Blueprint, render_template, request, session, redirect, url_for
from services.symptom import (
    clean_symptom_input,
    ai_predict_disease,
    ai_explain_disease,
    is_valid_symptom_input
)
from utils.auth import login_required
from utils.activity_logger import log_activity

symptom_bp = Blueprint("symptom", __name__)

@symptom_bp.route("/symptom", methods=["GET", "POST"])
@login_required
def symptom_page():

    # ---------- POST ----------
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

        # ✅ AI CALLS ONLY HERE
        disease = ai_predict_disease(cleaned)
        explanation = ai_explain_disease(cleaned, disease)

        # ✅ Save to session
        session["last_symptom_text"] = symptoms
        session["last_symptom_disease"] = disease
        session["last_symptom_explanation"] = explanation

        # ✅ Save to DB ONCE
        log_activity(
            module="symptom",
            user_input=symptoms,
            ai_output=f"Disease: {disease}\n\n{explanation}"
        )

        return redirect(url_for("symptom.symptom_page"))

    # ---------- GET ----------
    return render_template(
        "symptom.html",
        symptoms=session.get("last_symptom_text", ""),
        disease=session.get("last_symptom_disease"),
        explanation=session.get("last_symptom_explanation")
    )