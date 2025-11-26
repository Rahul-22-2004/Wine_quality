# Main.py - ENOLOGIX: Fully MongoDB Atlas Powered (2025)
from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask_pymongo import PyMongo
from dotenv import load_dotenv
from bson.objectid import ObjectId
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import qrcode
import io

# ==================== APP SETUP ====================
app = Flask(__name__)
load_dotenv()

app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-only-for-local")
app.config["MONGO_URI"] = os.getenv("MONGO_URI")

mongo = PyMongo(app)

# ----------- CRITICAL CHECK (adds clear error message) -----------
if not app.config.get("MONGO_URI"):
    raise RuntimeError("MONGO_URI is missing! Add it to your .env file.")

with app.app_context():
    try:
        # This line forces the connection and will raise an error immediately if wrong
        mongo.cx.server_info()          # ← tests the connection right now
        print("MongoDB Atlas connected successfully!")
        # Force creation of the database/collections (they are created on first insert anyway)
        mongo.db.command("ping")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to MongoDB Atlas!\nError: {e}\n\n"
                           "Check these common issues:\n"
                           "1. MONGO_URI in .env is correct (copy-paste from Atlas)\n"
                           "2. Your IP is added to Network Access (or use 0.0.0.0/0 for testing)\n"
                           "3. Username/password in URI are URL-encoded if they contain special chars\n"
                           "4. Atlas cluster is not paused") from e
# -----------------------------------------------------------------

# ==================== MODEL LOADING ====================
try:
    with open('wine_quality_rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    try:
        with open('wine_quality_gb_model.pkl', 'rb') as f:
            gb_model = pickle.load(f)
    except FileNotFoundError:
        gb_model = rf_model

    with open('wine_quality_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('wine_quality_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    print("All ML models loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    rf_model = gb_model = scaler = feature_names = None

# ==================== DECORATORS ====================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or not user.get('is_admin', False):
            flash('Access Denied: Admin privileges required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated

# ==================== HELPER FUNCTIONS ====================
def create_features(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                    density, pH, sulphates, alcohol):
    features = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
    }
    features['acidity_ratio'] = fixed_acidity / (volatile_acidity + 0.001)
    features['sulfur_ratio'] = free_sulfur_dioxide / (total_sulfur_dioxide + 0.001)
    features['acid_sugar_interaction'] = citric_acid * residual_sugar
    features['alcohol_density'] = alcohol / density
    features['sulphates_alcohol'] = sulphates * alcohol
    features['total_acidity'] = fixed_acidity + volatile_acidity + citric_acid
    features['bound_sulfur'] = total_sulfur_dioxide - free_sulfur_dioxide
    return features

def predict_wine_type(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                      density, pH, sulphates, alcohol):
    red_score = white_score = 0
    if total_sulfur_dioxide > 140: white_score += 3
    elif total_sulfur_dioxide > 100: white_score += 2
    elif total_sulfur_dioxide < 60: red_score += 2
    else: red_score += 1
    if free_sulfur_dioxide > 40: white_score += 2
    elif free_sulfur_dioxide < 20: red_score += 1
    if citric_acid > 0.4: white_score += 2
    elif citric_acid < 0.25: red_score += 2
    if pH < 3.2: white_score += 1
    elif pH > 3.4: red_score += 1
    if density > 0.9975: red_score += 1
    elif density < 0.9950: white_score += 1
    if fixed_acidity > 8.0: red_score += 1
    elif fixed_acidity < 7.0: white_score += 1
    if volatile_acidity > 0.5: red_score += 2
    elif volatile_acidity < 0.3: white_score += 1
    if chlorides > 0.09: red_score += 1
    elif chlorides < 0.05: white_score += 1
    if residual_sugar > 10: white_score += 2
    elif residual_sugar > 5: white_score += 1

    total = red_score + white_score or 1
    if red_score > white_score:
        return "Red Wine", round((red_score / total) * 100, 1)
    else:
        return "White Wine", round((white_score / total) * 100, 1)

def calculate_wine_price(quality, alcohol, volatile_acidity, residual_sugar, pH, sulphates, wine_type):
    import random
    base = (quality ** 1.8) * 3
    factors = (
        (alcohol - 8) * 2.5 if alcohol > 8 else 0 +
        (residual_sugar * 0.3 if residual_sugar > 10 else 0) +
        (sulphates - 0.5) * 3 if sulphates > 0.5 else 0 +
        abs(3.3 - pH) * -2 -
        volatile_acidity * 8
    )
    price = max((base + factors) * (1.2 if wine_type == "Red Wine" else 1.0), 5.0)
    return round(price * random.uniform(0.95, 1.05), 2)

# ==================== ROUTES ====================
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else 'login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        firstname = request.form['firstname'].strip()
        lastname = request.form['lastname'].strip()
        email = request.form['email'].strip().lower()
        phone = request.form['phone'].strip()
        password = request.form['password']
        confirm = request.form['confirm_password']

        if password != confirm:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')

        if mongo.db.users.find_one({'$or': [{'username': username}, {'email': email}]}):
            flash('Username or email already exists!', 'error')
            return render_template('register.html')

        mongo.db.users.insert_one({
            'username': username,
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'phone': phone,
            'password': generate_password_hash(password),
            'is_admin': False,
            'theme': 'dark',
            'model_preference': 'rf',
            'email_notif': False,
            'push_notif': False,
            'created_at': datetime.utcnow()
        })
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        
        user = mongo.db.users.find_one({'username': username})
        
        if not user:
            flash('Invalid username or password', 'error')
            return render_template('login.html')
            
        if check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['is_admin'] = user.get('is_admin', False)
            session['model_preference'] = user.get('model_preference', 'rf')
            flash(f'Welcome back, {user["firstname"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('dashboard.html', is_admin=user.get('is_admin', False))

@app.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html')

@app.route('/performance')
@login_required
def performance():
    # Optional: pass some stats if you want
    user_stats = {
        'total': mongo.db.history.count_documents({'user_id': session['user_id']}),
        'avg_quality': round(
            mongo.db.history.aggregate([
                {'$match': {'user_id': session['user_id']}},
                {'$group': {'_id': None, 'avg': {'$avg': '$quality'}}}
            ]).__next__().get('avg', 0) if mongo.db.history.find_one({'user_id': session['user_id']}) else 0,
            1
        )
    }
    return render_template('performance.html', stats=user_stats)

@app.route('/result', methods=['POST'])
@login_required
def result():
    if not rf_model:
        flash('ML Model not available. Contact admin.', 'error')
        return redirect(url_for(prediction))

    try:
        # Parse form
        inputs = {k: float(request.form[k]) for k in [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
            'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
            'pH', 'sulphates', 'alcohol'
        ]}

        # Predict
        features_df = pd.DataFrame([create_features(**inputs)])
        features_df = features_df[feature_names]
        scaled = scaler.transform(features_df)
        model = rf_model if session.get('model_preference', 'rf') == 'rf' else gb_model
        quality = int(model.predict(scaled)[0])

        wine_type, confidence = predict_wine_type(**inputs)
        price_usd = calculate_wine_price(quality, inputs['alcohol'], inputs['volatile_acidity'],
                                        inputs['residual_sugar'], inputs['pH'], inputs['sulphates'], wine_type)
        price_inr = round(price_usd * 83, 2)

        # Save to MongoDB
        mongo.db.history.insert_one({
            **inputs,
            'user_id': session['user_id'],
            'wine_type': wine_type,
            'quality': quality,
            'price_usd': f"${price_usd:.2f}",
            'price_inr': f"₹{price_inr:.2f}",
            'confidence': confidence,
            'model_used': session.get('model_preference', 'rf'),
            'created_at': datetime.utcnow()
        })

        return render_template('result.html',
                               predicted_quality=quality,
                               estimated_price_usd=f"${price_usd:.2f}",
                               estimated_price_inr=f"₹{price_inr:.2f}",
                               wine_type=wine_type,
                               wine_confidence=f"{confidence}%")

    except Exception as e:
        flash(f'Prediction failed: {str(e)}', 'error')
        return redirect(url_for('prediction'))

@app.route('/history')
@login_required
def history():
    history = list(mongo.db.history.find({'user_id': session['user_id']}).sort('created_at', -1))
    for entry in history:
        entry['id'] = str(entry['_id'])
        entry['created_at'] = entry['created_at'].strftime('%Y-%m-%d %H:%M')
    return render_template('history.html', history=history)

@app.route('/delete_history/<id>', methods=['POST'])
@login_required
def delete_history(id):
    result = mongo.db.history.delete_one({
        '_id': ObjectId(id),
        'user_id': session['user_id']
    })
    return jsonify(success=result.deleted_count > 0)

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    mongo.db.history.delete_many({'user_id': session['user_id']})
    return jsonify(success=True)

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    data = request.get_json()
    mongo.db.users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': {
            'theme': data.get('theme', 'dark'),
            'model_preference': data.get('model', 'rf'),
            'email_notif': data.get('emailNotif', False),
            'push_notif': data.get('pushNotif', False)
        }}
    )
    session['model_preference'] = data.get('model', 'rf')
    return jsonify(success=True)

# ==================== ADMIN ROUTES ====================
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_users = mongo.db.users.count_documents({})
    total_predictions = mongo.db.history.count_documents({})
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_users = mongo.db.users.count_documents({'created_at': {'$gte': today}})

    recent_users = list(mongo.db.users.find().sort('created_at', -1).limit(10))
    admins = list(mongo.db.users.find({'is_admin': True}))

    for u in recent_users + admins:
        u['id'] = str(u['_id'])

    return render_template('admin_dashboard.html',
                           total_users=total_users,
                           today_users=today_users,
                           recent_users=recent_users,
                           total_predictions=total_predictions,
                           all_admins=admins,
                           current_user_id=session['user_id'])

@app.route('/admin/make_admin', methods=['POST'])
@admin_required
def make_admin():
    email = request.form.get('email', '').strip().lower()
    if not email:
        flash('Email required', 'error')
        return redirect(url_for('admin_dashboard'))

    result = mongo.db.users.update_one(
        {'email': email},
        {'$set': {'is_admin': True}}
    )
    if result.modified_count:
        flash(f'Admin rights granted to {email}', 'success')
    else:
        flash('User not found', 'error')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/remove_admin', methods=['POST'])
@admin_required
def remove_admin():
    user_id = request.form.get('user_id')
    if user_id == session['user_id']:
        flash('Cannot remove yourself', 'error')
        return redirect(url_for('admin_dashboard'))

    mongo.db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'is_admin': False}}
    )
    flash('Admin rights removed', 'success')
    return redirect(url_for('admin_dashboard'))

# ==================== PUBLIC & API ====================
@app.route('/stats')
def public_stats():
    return render_template('public_stats.html',
                           total_users=mongo.db.users.count_documents({}),
                           total_predictions=mongo.db.history.count_documents({}))

@app.route('/api/user_stats')
@login_required
def user_stats_api():
    pipeline = [
        {'$match': {'user_id': session['user_id']}},
        {'$group': {'_id': None, 'total': {'$sum': 1}, 'avg': {'$avg': '$quality'}}}
    ]
    result = list(mongo.db.history.aggregate(pipeline))
    data = result[0] if result else {'total': 0, 'avg': 0}
    return jsonify({
        'total_predictions': data['total'],
        'avg_quality': round(data['avg'] or 0, 1)
    })

# ==================== QR CODE ====================
@app.route('/generate_qr')
def generate_qr():
    url = request.host_url.rstrip('/')
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png', as_attachment=True, download_name='enologix_qr.png')

@app.route('/qr')
def qr_page():
    return render_template('qr_code.html', app_url=request.host_url.rstrip('/'))

@app.route('/test-db')
def test_db():
    try:
        mongo.cx.server_info()
        return "<h1>MongoDB Atlas Connected Successfully!</h1>"
    except Exception as e:
        return f"<h1>Connection Failed:</h1><pre>{e}</pre>"

# ==================== RUN ====================
if __name__ == '__main__':
    print("\n" + "="*70)
    print(" ENOLOGIX - Running on MongoDB Atlas")
    print(" All SQLite code removed | Fully cloud-ready")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)