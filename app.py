print("✅ MAIN APP.PY IS RUNNING")

from flask import Flask, render_template
from database import db



# -------------------
# CREATE APP (ONCE)
# -------------------
app = Flask(__name__)
app.secret_key = "medisense_super_secret_key_123"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///medisense.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# -------------------
# INIT DATABASE
# -------------------
db.init_app(app)

with app.app_context():
    import models
    db.create_all()

# -------------------
# REGISTER BLUEPRINTS
# -------------------
from routes.auth import auth_bp
from routes.symptom import symptom_bp
from routes.mental import mental_bp
from routes.pdf import pdf_bp
from routes.drug import drug_bp
from routes.preventive import preventive_bp

app.register_blueprint(auth_bp)
app.register_blueprint(symptom_bp)
app.register_blueprint(mental_bp)
app.register_blueprint(pdf_bp)
app.register_blueprint(drug_bp)
app.register_blueprint(preventive_bp, url_prefix="/preventive")


from flask import render_template, session, redirect, url_for

@app.route("/")
def root():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("auth.login"))
# -------------------
# ROUTES
# -------------------
@app.route("/home")
def home():
    return render_template("home.html")

# -------------------
# RUN
# -------------------
if __name__ == "__main__":
    app.run(debug=True)