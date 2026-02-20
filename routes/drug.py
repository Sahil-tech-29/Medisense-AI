from flask import Blueprint, render_template, request, session, redirect, url_for
from services.drug import (
    analyze_drug_review,
    generate_drug_risk_explanation
)

drug_bp = Blueprint("drug", __name__)

@drug_bp.route("/drug", methods=["GET", "POST"])
def drug_page():

    # ---------- GET ----------
    if request.method == "GET":

        # ❌ If this GET is from refresh / reopen → clear everything
        if not session.pop("drug_just_predicted", False):
            session.pop("last_drug_review", None)
            session.pop("last_drug_result", None)
            session.pop("last_drug_explanation", None)

        return render_template(
            "drug.html",
            review=session.get("last_drug_review", ""),
            result=session.get("last_drug_result"),
            explanation=session.get("last_drug_explanation"),
            error=session.pop("drug_error", None)
        )

    # ---------- POST ----------
    review = request.form.get("review", "").strip()

    if len(review.split()) < 5:
        session["drug_error"] = "Please describe your experience in a little more detail."
        return redirect(url_for("drug.drug_page"))

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

    # ✅ mark that GET after this is allowed to show data
    session["drug_just_predicted"] = True

    return redirect(url_for("drug.drug_page"))
