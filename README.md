# 🧠 Medisense — AI-Powered Healthcare Awareness Platform

Medisense is a **full-stack AI healthcare awareness platform** designed to help users understand health concerns through **AI-assisted insights**, while maintaining a **non-diagnostic, educational, and ethical approach**.

It integrates **Machine Learning, Generative AI, secure authentication, session management, and persistent data storage** into a clean, modern web experience.

---

## 🚀 Key Features

### 🩺 1. Symptom-Based Disease Awareness
- Users describe symptoms in natural language
- AI predicts the **most likely condition**
- Provides **structured, easy-to-understand explanation**
- Handles invalid / meaningless inputs safely
- Deterministic behavior for same input
- Full activity logging per user

---

### 🧠 2. Mental Health Awareness
- Detects emotional patterns (stress, anxiety, low mood, etc.)
- Uses **empathetic, non-clinical AI explanations**
- Input validation to prevent meaningless predictions
- Consistent predictions for repeated input
- Designed for **support, not diagnosis**

---

### 📄 3. Medical PDF Report Explanation
- Upload digital medical reports (PDF)
- Extracts readable medical text
- Generates **structured AI explanations**
- Separate views for:
  - AI explanation
  - Raw extracted report text
- Detects non-medical / scanned PDFs
- Clean session handling (no stale results)

---

### 💊 4. Drug Side-Effect Risk Analysis
- Analyzes user-reported medicine experiences
- Detects:
  - Critical symptoms
  - Allergic reactions
  - Concerning patterns
  - Mild side effects
- Uses **rule-based safety overrides**
- Produces detailed AI explanations
- Realistic risk assessment logic
- Input unpredictability handled safely

---

### 🌿 5. Preventive Health Risk Awareness
- Lifestyle-based risk assessment:
  - Activity
  - Diet
  - Sleep
  - Stress
  - Smoking & Alcohol
- Visual risk indicators & confidence bars
- Structured AI preventive guidance
- Separate result page for better UX
- No diagnosis, no prescriptions — awareness only

---

## 🔐 Authentication & Security

- Secure user registration & login
- Password hashing using `werkzeug`
- Session-based authentication
- Protected routes (`login_required`)
- Case-insensitive usernames (no duplicate accounts)
- Logout & session cleanup

---

## 🗃️ Data Persistence & Activity Tracking

- SQLite database using **Flask-SQLAlchemy**
- Stores:
  - User accounts
  - User activities across all modules
- Centralized activity logger
- Ready for dashboards & analytics expansion

---

## 🧠 AI & ML Stack

- **Groq LLM API** for fast AI inference
- Prompt-engineered structured outputs
- Deterministic logic layered on top of AI
- Safety-first design:
  - No medical diagnosis
  - No prescriptions
  - No alarming language

---

## 🧱 Tech Stack

| Layer | Technology |
|-----|------------|
| Backend | Flask |
| Frontend | HTML, CSS |
| AI | Groq LLM |
| ML | TensorFlow, Scikit-Learn |
| Database | SQLite |
| ORM | Flask-SQLAlchemy |
| Auth | Sessions + Password Hashing |

---

## 📂 Project Structure
