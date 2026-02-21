from flask import Blueprint, render_template, request, session, redirect, url_for
from services.drug import analyze_drug_review, generate_drug_risk_explanation
from utils.activity_logger import log_activity
from utils.auth import login_required

drug_bp = Blueprint("drug", __name__)

@drug_bp.route("/drug", methods=["GET", "POST"])
@login_required
def drug_page():

    # ---------- POST ----------
    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if len(review.split()) < 5:
            session["drug_error"] = "Please describe your experience in a little more detail."
            session["last_drug_review"] = review
            return redirect(url_for("drug.drug_page"))

        # 🔁 reuse cached result
        if session.get("last_drug_review") != review:
            result = analyze_drug_review(review)
            explanation = generate_drug_risk_explanation(
                review,
                result["risk_label"],
                result["confidence"],
                result["effects"]
            )

            session["last_drug_review"] = review
            session["last_drug_result"] = result
            session["last_drug_explanation"] = explanation

            # ✅ LOG ONCE
            log_activity(
                module="drug",
                user_input=review,
                ai_output=f"{result['risk_label']} ({result['confidence']}%)\n\n{explanation}"
            )

        return redirect(url_for("drug.drug_page"))

    # ---------- GET ----------
    return render_template(
        "drug.html",
        review=session.get("last_drug_review", ""),
        result=session.get("last_drug_result"),
        explanation=session.get("last_drug_explanation"),
        error=session.pop("drug_error", None)
    )


# ---------- CLEAR ROUTE ----------
@drug_bp.route("/drug/reset")
@login_required
def reset_drug():
    session.pop("last_drug_review", None)
    session.pop("last_drug_result", None)
    session.pop("last_drug_explanation", None)
    session.pop("drug_error", None)
    return redirect(url_for("drug.drug_page"))