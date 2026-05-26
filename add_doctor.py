from app import app
from auth import db, bcrypt, Doctor

with app.app_context():
    # Check if this doctor already exists
    existing = Doctor.query.filter_by(email="doctor@hospital.org").first()
    if existing:
        print(f"Doctor already exists: {existing.name} ({existing.email})")
    else:
        # Hash passcode
        hashed_password = bcrypt.generate_password_hash("password123").decode('utf-8')
        
        # Create doctor
        doctor = Doctor(
            name="Akash",
            email="doctor@hospital.org",
            license_number="MD-9999-X",
            password=hashed_password,
            hospital="Metropolis Advanced Imaging Labs"
        )
        
        db.session.add(doctor)
        db.session.commit()
        print("Successfully registered default doctor:")
        print(" - Name: Dr. Akash")
        print(" - Email: doctor@hospital.org")
        print(" - Password: password123")
        print(" - License: MD-9999-X")
