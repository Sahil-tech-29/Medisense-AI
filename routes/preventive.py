from flask import Blueprint, render_template, request, redirect, url_for, session
from services.preventive import assess_preventive_health

preventive_bp = Blueprint("preventive", __name__)

preventive_results=None


@preventive_bp.route("/", methods=["GET", "POST"])
def preventive_page():
    if request.method == "POST":

        age = request.form.get("age")
        activity = request.form.get("activity")
        diet = request.form.get("diet")
        sleep = request.form.get("sleep")
        stress = request.form.get("stress")
        smoking = request.form.get("smoking")
        alcohol = request.form.get("alcohol")

        if not all([age, activity, diet, sleep, stress, smoking, alcohol]):
            session["preventive_error"] = (
                "Please fill in all lifestyle details before assessment."
            )
            return redirect(url_for("preventive.preventive_page"))

        inputs = {
            "age": int(age),
            "activity": activity,
            "diet": diet,
            "sleep": int(sleep),
            "stress": int(stress),
            "smoking": smoking,
            "alcohol": alcohol
        }
        
        session["preventive_result"] = assess_preventive_health(inputs)
        global preventive_results
        preventive_results=assess_preventive_health(inputs)
        return redirect(url_for("preventive.preventive_result"))


    
    error = session.pop("preventive_error", None)
    return render_template("preventive.html", error=error)


@preventive_bp.route("/result", methods=["GET"])
def preventive_result():
    # result = session.get("preventive_result")

    # if not result:
    #     return redirect(url_for("preventive.preventive_page"))

    result=preventive_results

    return render_template("preventive_result.html", result=result)