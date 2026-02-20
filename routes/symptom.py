from flask import Blueprint, render_template, request ,session,redirect ,url_for
from services.symptom import (
    clean_symptom_input,
    ai_predict_disease,
    ai_explain_disease,
    is_valid_symptom_input
)

symptom_bp = Blueprint("symptom", __name__)

@symptom_bp.route("/symptom", methods=["GET", "POST"])
def symptom_page():

    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()

        if symptoms:
            cleaned = clean_symptom_input(symptoms)

            if not is_valid_symptom_input(cleaned):
                session["last_symptom_disease"] = "Input not understood"
                session["last_symptom_explanation"] = (
                    "<p>Please enter meaningful symptoms like fever, pain, cough, or headache.</p>"
                )
                session["last_symptom_text"] = symptoms
                return redirect(url_for("symptom.symptom_page"))

            disease = ai_predict_disease(cleaned)
            explanation = ai_explain_disease(cleaned, disease)

            session["last_symptom_text"] = cleaned
            session["last_symptom_disease"] = disease
            session["last_symptom_explanation"] = explanation

        return redirect(url_for("symptom.symptom_page"))


    # ---------- GET ----------
    disease = session.pop("last_symptom_disease", None)
    explanation = session.pop("last_symptom_explanation", None)
    symptoms = session.pop("last_symptom_text", "")

    return render_template(
        "symptom.html",
        symptoms=symptoms,
        disease=disease,
        explanation=explanation
    )
