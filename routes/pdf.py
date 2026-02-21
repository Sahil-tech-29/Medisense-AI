from flask import Blueprint, render_template, request, redirect, url_for, session
from services.pdf import process_medical_pdf
from utils.activity_logger import log_activity
from utils.auth import login_required

pdf_bp = Blueprint("pdf", __name__)

@pdf_bp.route("/pdf", methods=["GET", "POST"])
@login_required
def pdf_page():

    # ---------- POST ----------
    if request.method == "POST":
        file = request.files.get("pdf")

        if not file:
            session["pdf_error"] = "Please upload a PDF file."
            return redirect(url_for("pdf.pdf_page"))

        output = process_medical_pdf(file)

        if "error" in output:
            session["pdf_error"] = output["error"]
            return redirect(url_for("pdf.pdf_page"))

        # ✅ Save result
        session["pdf_text"] = output["text"]
        session["pdf_summary"] = output["summary"]
        session["pdf_filename"] = file.filename

        # ✅ LOG ONCE (POST ONLY)
        log_activity(
            module="medical_pdf",
            user_input=file.filename,
            ai_output=output["summary"]
        )

        return redirect(url_for("pdf.pdf_page"))

    # ---------- GET ----------
    return render_template(
        "pdf.html",
        extracted_text=session.get("pdf_text"),
        summary=session.get("pdf_summary"),
        filename=session.get("pdf_filename"),
        error=session.pop("pdf_error", None)
    )


# ---------- RESET ROUTE ----------
@pdf_bp.route("/pdf/reset")
@login_required
def reset_pdf():
    session.pop("pdf_text", None)
    session.pop("pdf_summary", None)
    session.pop("pdf_filename", None)
    session.pop("pdf_error", None)
    return redirect(url_for("pdf.pdf_page"))