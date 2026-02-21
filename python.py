from app import app
from models import UserActivity

with app.app_context():
    data = UserActivity.query.all()
    for d in data:
        print(d.module, d.user_input)