from database import db
from models import UserActivity
from flask import session

def log_activity(module, user_input, ai_output):
    if "user_id" not in session:
        return

    activity = UserActivity(
        user_id=session["user_id"],
        module=module,
        user_input=user_input,
        ai_output=ai_output
    )

    db.session.add(activity)
    db.session.commit()