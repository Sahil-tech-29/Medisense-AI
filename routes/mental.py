from flask import Blueprint, render_template, request, session, redirect, url_for
from services.mental_health import final_prediction
from services.mental_health import generate_mental_health_explanation
from utils.auth import login_required
from utils.activity_logger import log_activity

mental_bp = Blueprint("mental", __name__)


@mental_bp.route("/mental", methods=["GET", "POST"])
@login_required
def mental_page():

    # ---------- POST ----------
    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            return redirect(url_for("mental.mental_page"))

        # 🔥 Basic validation (same style as symptom)
        if len(text) < 10 or len(text.split()) < 3:
            session["last_mental_text"] = text
            session["last_mental_label"] = "Input not understood"
            session["last_mental_explanation"] = (
                "<p>Please describe your feelings in more detail.</p>"
            )
            return redirect(url_for("mental.mental_page"))

        # 🔥 AI PIPELINE (same as symptom)
        label, confidence, top3 = final_prediction(text)

        explanation = generate_mental_health_explanation(text, label)

        # Save to session
        session["last_mental_text"] = text
        session["last_mental_label"] = str(label)
        session["last_mental_confidence"] = float(confidence)
        session["last_mental_explanation"] = str(explanation)
        session["mental_top3"] = top3 if top3 else []

        # Save to DB
        log_activity(
            module="mental_health",
            user_input=text,
            ai_output=f"State: {label}\n\n{explanation}"
        )

        return redirect(url_for("mental.mental_page"))

    # ---------- GET ----------
    return render_template(
        "mental.html",
        text=session.get("last_mental_text", ""),
        label=session.get("last_mental_label"),
        explanation=session.get("last_mental_explanation"),
        confidence=session.get("last_mental_confidence"),
        top3=session.get("mental_top3")
    )