"""
PACS Doctor Authentication Blueprint
Author: Antigravity AI
Version: 1.0.0
Description: Provides database modeling for clinical personnel (Doctors) using SQLAlchemy,
             hashes passwords with Bcrypt, manages sessions using Flask-Login, and defines
             routes for user registration, authentication, and secure session terminations.
"""

from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, login_user, logout_user, login_required, current_user

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Setup blueprint
auth_bp = Blueprint('auth', __name__)

# Extensions instances (to be initialized with init_app inside app.py)
db = SQLAlchemy()
bcrypt = Bcrypt()
limiter = Limiter(key_func=get_remote_address, default_limits=["300 per day", "100 per hour"])

# ---------------------------------------------------------
# Doctor Database Model Table
# ---------------------------------------------------------
import json

class Doctor(UserMixin, db.Model):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    hospital = db.Column(db.String(100), default='Metropolis Imaging Lab')
    
    # Establish one-to-many relationship with clinical scans
    scans = db.relationship('Scan', backref='doctor', lazy=True, cascade="all, delete-orphan")

    def __repr__(self):
        return f'<Doctor: {self.name} (License: {self.license_number})>'

class Scan(db.Model):
    __tablename__ = 'scans'

    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.String(50), unique=True, nullable=False)
    patient_id = db.Column(db.String(50), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.String(10), nullable=False)
    patient_gender = db.Column(db.String(10), nullable=False)
    referred_by = db.Column(db.String(100), nullable=False)
    original_image_url = db.Column(db.String(255), nullable=False)
    heatmap_image_url = db.Column(db.String(255), nullable=False)
    
    # AI Classifier Probabilities (%)
    metric_normal = db.Column(db.Float, nullable=False)
    metric_pneumonia = db.Column(db.Float, nullable=False)
    metric_cardiomegaly = db.Column(db.Float, nullable=False)
    metric_effusion = db.Column(db.Float, nullable=False)
    metric_pneumothorax = db.Column(db.Float, nullable=False)

    # Heuristic clinical features
    heuristic_ctr = db.Column(db.Float, nullable=False)
    heuristic_density = db.Column(db.Float, nullable=False)
    heuristic_asymmetry = db.Column(db.Float, nullable=False)
    heuristic_costo = db.Column(db.Float, nullable=False)
    heuristic_apical = db.Column(db.Float, nullable=False)

    dominant_finding = db.Column(db.String(100), nullable=False)
    dominant_score = db.Column(db.Float, nullable=False)
    impressions_json = db.Column(db.Text, nullable=False)  # JSON-encoded clinical narrative list
    timestamp = db.Column(db.String(50), nullable=False)

    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)

    @property
    def impressions(self):
        try:
            return json.loads(self.impressions_json)
        except Exception:
            return []

    @impressions.setter
    def impressions(self, value):
        self.impressions_json = json.dumps(value)

    def to_dict(self):
        """Serializes scan record for secure API responses."""
        return {
            "success": True,
            "scan_id": self.scan_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "referred_by": self.referred_by,
            "timestamp": self.timestamp,
            "original_image_url": self.original_image_url,
            "heatmap_image_url": self.heatmap_image_url,
            "metrics": {
                "cardiomegaly": self.metric_cardiomegaly,
                "pneumonia": self.metric_pneumonia,
                "effusion": self.metric_effusion,
                "pneumothorax": self.metric_pneumothorax,
                "normal": self.metric_normal
            },
            "heuristics": {
                "ctr": self.heuristic_ctr,
                "lung_density": self.heuristic_density,
                "lung_asymmetry": self.heuristic_asymmetry,
                "costophrenic_density": self.heuristic_costo,
                "apical_flatness": self.heuristic_apical
            },
            "dominant_finding": self.dominant_finding,
            "dominant_score": self.dominant_score,
            "impressions": self.impressions
        }

# ---------------------------------------------------------
# Authentication Routes
# ---------------------------------------------------------
@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    """Handles secure doctor login sessions."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Security Error: Credentials payload cannot be empty.', 'danger')
            return render_template('login.html')
            
        # Lookup user in db
        doctor = Doctor.query.filter_by(email=email).first()
        
        # Verify credentials
        if doctor and bcrypt.check_password_hash(doctor.password, password):
            login_user(doctor)
            flash(f"Access Granted. Session initialized for Dr. {doctor.name}.", "success")
            return redirect(url_for('index'))
        else:
            flash('Access Denied. Invalid email or security passcode.', 'danger')
            
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def register():
    """Registers new clinical personnel after validating licensing."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        license_number = request.form.get('license_number', '').strip()
        password = request.form.get('password', '')
        hospital = request.form.get('hospital', '').strip() or 'Metropolis Imaging Lab'
        
        if not name or not email or not license_number or not password:
            flash('Registration Error: All highlighted fields are mandatory.', 'danger')
            return render_template('register.html')

        # Enforce clinical passcode complexity
        import re
        if (len(password) < 8 or 
            not re.search("[a-z]", password) or 
            not re.search("[A-Z]", password) or 
            not re.search("[0-9]", password) or 
            not re.search("[_@$!%*#?&\\-]", password)):
            flash('Security Error: Passcode must be at least 8 characters long and contain uppercase, lowercase, numbers, and a special character.', 'danger')
            return render_template('register.html')
            
        # Verify if email or license number is already registered
        if Doctor.query.filter_by(email=email).first():
            flash('Conflict: This email address is already registered in the PACS system.', 'danger')
            return render_template('register.html')
            
        if Doctor.query.filter_by(license_number=license_number).first():
            flash('Conflict: This license number is already registered in the PACS system.', 'danger')
            return render_template('register.html')
            
        # Hash password securely
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        
        # Create database entry
        new_doctor = Doctor(
            name=name,
            email=email,
            license_number=license_number,
            password=hashed_password,
            hospital=hospital
        )
        
        try:
            db.session.add(new_doctor)
            db.session.commit()
            flash('Licensing verified. PACS Credentials generated successfully. Please login.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Database Error: Failed to commit records. Details: {e}', 'danger')
            
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    """Terminates active doctor sessions securely."""
    name = current_user.name
    logout_user()
    flash(f"Session terminated safely for Dr. {name}.", "success")
    return redirect(url_for('auth.login'))
