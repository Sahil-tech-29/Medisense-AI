from flask import Flask, render_template
from routes.symptom import symptom_bp
from routes.mental import mental_bp
from routes.pdf import pdf_bp
from routes.drug import drug_bp
from routes.preventive import preventive_bp





app = Flask(__name__)
app.secret_key = "medisense_super_secret_key_123"


app.register_blueprint(mental_bp)
app.register_blueprint(symptom_bp)
app.register_blueprint(pdf_bp)
app.register_blueprint(drug_bp)
app.register_blueprint(preventive_bp)



@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
