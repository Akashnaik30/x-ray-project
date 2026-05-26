from app import app
from auth import db, Doctor

with app.app_context():
    doctors = Doctor.query.all()
    print(f"Total doctors in database: {len(doctors)}")
    for d in doctors:
        print(f"Name: {d.name}, Email: {d.email}")
