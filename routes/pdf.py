from flask import Blueprint, render_template, request, redirect, url_for, session
from services.pdf import process_medical_pdf

pdf_bp = Blueprint("pdf", __name__)

@pdf_bp.route("/pdf", methods=["GET", "POST"])
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
        else:
            session["pdf_text"] = output["text"]
            session["pdf_summary"] = output["summary"]
            session["pdf_filename"] = file.filename

        return redirect(url_for("pdf.pdf_page"))

    # ---------- GET (AUTO-CLEAR LIKE SYMPTOM) ----------
    extracted_text = session.pop("pdf_text", None)
    summary = session.pop("pdf_summary", None)
    filename = session.pop("pdf_filename", None)
    error = session.pop("pdf_error", None)

    return render_template(
        "pdf.html",
        extracted_text=extracted_text,
        summary=summary,
        filename=filename,
        error=error
    )