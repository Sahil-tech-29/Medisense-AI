from flask import Blueprint, render_template, request, session, redirect, url_for
from services.mental_health import generate_mental_health_explanation
from utils.groq_client import generate_with_groq

mental_bp = Blueprint("mental", __name__)

def predict_mental_label(text):
    prompt = f"""
Classify the user's emotional state into ONE category only.

Categories:
- Emotional Distress
- Anxiety Patterns
- Low Mood
- Stress Overload
- Generally Stable

User text:
{text}

Return ONLY the category name.
"""
    return generate_with_groq(prompt).strip()


@mental_bp.route("/mental", methods=["GET", "POST"])
def mental_page():

    # ---------- POST ----------
    if request.method == "POST":
        text = request.form.get("text", "").strip()

        # validation
        if not text or len(text) < 10 or len(text.split()) < 3:
            session["mental_error"] = "Please describe your feelings in a bit more detail."
            session["last_mental_text"] = text
            return redirect(url_for("mental.mental_page"))

        # reuse cached result
        if session.get("last_mental_text") != text:
            label = predict_mental_label(text)
            explanation = generate_mental_health_explanation(text, label)

            session["last_mental_text"] = text
            session["last_mental_label"] = label
            session["last_mental_explanation"] = explanation

        return redirect(url_for("mental.mental_page"))  # 🔑 KEY LINE

    # ---------- GET ----------
    text = session.pop("last_mental_text", "")
    label = session.pop("last_mental_label", None)
    explanation = session.pop("last_mental_explanation", None)
    error = session.pop("mental_error", None)


    from utils.activity_logger import log_activity

    log_activity(
        module="mental_health",
        user_input=text,
        ai_output=explanation
    )
    
    return render_template(
        "mental.html",
        text=text,
        label=label,
        explanation=explanation,
        error=error
    )