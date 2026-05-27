from flask import Blueprint, render_template, request, session, redirect, url_for
from services.drug import final_review_analysis
from utils.activity_logger import log_activity
from utils.auth import login_required

drug_bp = Blueprint("drug",__name__)

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

        # reuse cached result
        if session.get("last_drug_review") != review:

            result = final_review_analysis(review)

            session["last_drug_review"] = review
            session["last_drug_result"] = result

            # LOG
            log_activity(
                module="drug",
                user_input=review,
                ai_output=(
                    f"Sentiment: {result['sentiment']}\n"
                    f"Side Effects: {result['side_effects']}\n"
                    f"Severity: {result['severity']}\n"
                    f"Confidence: {result['confidence']:.2f}\n\n"
                    f"{result.get('explanation', '')}"
                )
            )

        return redirect(url_for("drug.drug_page"))

    # ---------- GET ----------
    return render_template(
        "drug.html",
        review=session.get("last_drug_review", ""),
        result=session.get("last_drug_result"),
        error=session.pop("drug_error", None)
    )

    # ---------- CLEAR ROUTE ----------

@drug_bp.route("/drug/reset")
@login_required
def reset_drug():
    session.pop("last_drug_review", None)
    session.pop("last_drug_result", None)
    session.pop("drug_error", None)
    return redirect(url_for("drug.drug_page"))
