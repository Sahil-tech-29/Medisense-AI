from flask import Blueprint, render_template, request, redirect, url_for, session
from models import User
from database import db

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    error = None

    if request.method == "POST":
        username = request.form.get("username").strip().lower()
        password = request.form.get("password")

        if User.query.filter_by(username=username).first():
            error = "Username already exists"
        else:
            user = User(username=username)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            return redirect(url_for("auth.login"))

    return render_template("register.html", error=error)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username").strip().lower()
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"

    return render_template("login.html", error=error)



@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))