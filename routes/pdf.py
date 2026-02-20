from flask import Blueprint, render_template, request, redirect, url_for, session
from services.pdf import process_medical_pdf

pdf_bp = Blueprint("pdf", __name__)

@pdf_bp.route("/pdf", methods=["GET", "POST"])
def pdf_page():

    # ---------- POST ----------
    if request.method == "POST":
        action = request.form.get("action")

        # ---- Upload PDF ----
        if action == "upload":
            file = request.files.get("pdf")

            if not file:
                session["pdf_error"] = "Please upload a PDF file."
                return redirect(url_for("pdf.pdf_page"))

            output = process_medical_pdf(file)

            if "error" in output:
                session["pdf_error"] = output["error"]
            else:
                session["summary"] = output["summary"]
                session["extracted_text"] = output["text"]
                session["pdf_filename"] = file.filename
                session["pdf_uploaded"] = True
                session["view_mode"] = None

            return redirect(url_for("pdf.pdf_page"))

        # ---- View AI ----
        if action == "view_ai":
            session["view_mode"] = "ai"
            return redirect(url_for("pdf.pdf_page"))

        # ---- View Text ----
        if action == "view_text":
            session["view_mode"] = "text"
            return redirect(url_for("pdf.pdf_page"))

    # ---------- GET ----------
    return render_template(
        "pdf.html",
        error=session.pop("pdf_error", None),
        summary=session.get("summary"),
        extracted_text=session.get("extracted_text"),
        view_mode=session.get("view_mode"),
        uploaded=session.get("pdf_uploaded"),
        filename=session.get("pdf_filename")
    )