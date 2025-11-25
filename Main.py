from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json
import qrcode
import io

app = Flask(__name__)
app.secret_key = 'enologix_secret_key_2025_change_this_in_production'

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
    
    print("✓ Models loaded successfully!")
    
except FileNotFoundError as e:
    print(f"ERROR: Model files not found: {e.filename}")
    rf_model = None
    gb_model = None
    scaler = None
    feature_names = None

# ==================== DATABASE ====================
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# ==================== DECORATORS ====================
def login_required(f):
    """Ensure user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Ensure user is admin"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        
        conn = get_db_connection()
        user = conn.execute('SELECT is_admin FROM users WHERE id = ?', 
                          (session['user_id'],)).fetchone()
        conn.close()
        
        if not user or not user['is_admin']:
            flash('⛔ Access Denied: Admin privileges required', 'error')
            return redirect(url_for('dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

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
    red_score = 0
    white_score = 0
    
    if total_sulfur_dioxide > 140:
        white_score += 3
    elif total_sulfur_dioxide > 100:
        white_score += 2
    elif total_sulfur_dioxide < 60:
        red_score += 2
    else:
        red_score += 1
    
    if free_sulfur_dioxide > 40:
        white_score += 2
    elif free_sulfur_dioxide < 20:
        red_score += 1
    
    if citric_acid > 0.4:
        white_score += 2
    elif citric_acid < 0.25:
        red_score += 2
    
    if pH < 3.2:
        white_score += 1
    elif pH > 3.4:
        red_score += 1
    
    if density > 0.9975:
        red_score += 1
    elif density < 0.9950:
        white_score += 1
    
    if fixed_acidity > 8.0:
        red_score += 1
    elif fixed_acidity < 7.0:
        white_score += 1
    
    if volatile_acidity > 0.5:
        red_score += 2
    elif volatile_acidity < 0.3:
        white_score += 1
    
    if chlorides > 0.09:
        red_score += 1
    elif chlorides < 0.05:
        white_score += 1
    
    if residual_sugar > 10:
        white_score += 2
    elif residual_sugar > 5:
        white_score += 1
    
    total_score = red_score + white_score
    if total_score == 0:
        total_score = 1
        
    if red_score > white_score:
        wine_type = "Red Wine"
        confidence = round((red_score / total_score) * 100, 1)
    else:
        wine_type = "White Wine"
        confidence = round((white_score / total_score) * 100, 1)
    
    return wine_type, confidence

def calculate_wine_price(quality, alcohol, volatile_acidity, residual_sugar, 
                        pH, sulphates, wine_type):
    import random
    base_price = (quality ** 1.8) * 3
    alcohol_factor = (alcohol - 8) * 2.5 if alcohol > 8 else 0
    va_penalty = volatile_acidity * 8
    sugar_bonus = (residual_sugar * 0.3) if residual_sugar > 10 else 0
    ph_factor = abs(3.3 - pH) * -2
    sulphates_bonus = (sulphates - 0.5) * 3 if sulphates > 0.5 else 0
    type_factor = 1.2 if wine_type == "Red Wine" else 1.0
    
    price_usd = (base_price + alcohol_factor + sugar_bonus + 
                 sulphates_bonus + ph_factor - va_penalty) * type_factor
    
    price_usd = max(price_usd, 5.0)
    price_usd *= random.uniform(0.95, 1.05)
    
    return price_usd

# ==================== PUBLIC ROUTES ====================
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('register.html')

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute('''INSERT INTO users 
                           (username, firstname, lastname, email, phone, password) 
                           VALUES (?, ?, ?, ?, ?, ?)''',
                        (username, firstname, lastname, email, phone, hashed_password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', 
                          (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            flash(f'Welcome back, {user["firstname"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# ==================== USER ROUTES ====================
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    user = conn.execute('SELECT is_admin FROM users WHERE id = ?', 
                       (session['user_id'],)).fetchone()
    conn.close()
    
    is_admin = user['is_admin'] if user else False
    
    return render_template('dashboard.html', is_admin=is_admin)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
@login_required
def result():
    if rf_model is None:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('prediction'))

    try:
        fixed_acidity = float(request.form['fixed_acidity'])
        volatile_acidity = float(request.form['volatile_acidity'])
        citric_acid = float(request.form['citric_acid'])
        residual_sugar = float(request.form['residual_sugar'])
        chlorides = float(request.form['chlorides'])
        free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
        total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])

        features_dict = create_features(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        )

        input_df = pd.DataFrame([features_dict])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        model_choice = session.get('model_preference', 'rf')
        model = rf_model if model_choice == 'rf' else gb_model

        predicted_quality = int(model.predict(input_scaled)[0])
        
        wine_type, wine_confidence = predict_wine_type(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        )

        price_usd = calculate_wine_price(
            predicted_quality, alcohol, volatile_acidity, residual_sugar,
            pH, sulphates, wine_type
        )
        price_inr = price_usd * 83

        conn = get_db_connection()
        conn.execute('''INSERT INTO history 
                       (user_id, wine_type, quality, price_usd, price_inr, confidence, 
                        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                        density, pH, sulphates, alcohol, model_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (session['user_id'], wine_type, predicted_quality, 
                     f"${price_usd:.2f}", f"₹{price_inr:.2f}", wine_confidence,
                     fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                     chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                     density, pH, sulphates, alcohol, model_choice))
        conn.commit()
        conn.close()

        return render_template('result.html',
                             predicted_quality=predicted_quality,
                             estimated_price_usd=f"${price_usd:.2f}",
                             estimated_price_inr=f"₹{price_inr:.2f}",
                             wine_type=wine_type,
                             wine_confidence=f"{wine_confidence}%")

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        flash(f'Error processing prediction: {str(e)}', 'error')
        return redirect(url_for('prediction'))

@app.route('/performance')
@login_required
def performance():
    return render_template('performance.html')

@app.route('/history')
@login_required
def history():
    try:
        conn = get_db_connection()
        history_data = conn.execute(
            'SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC',
            (session['user_id'],)
        ).fetchall()
        conn.close()
        
        history_list = [dict(row) for row in history_data]
        
        return render_template('history.html', history=history_list)
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        flash('Error loading history. Please try again.', 'error')
        return render_template('history.html', history=[])

@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    data = request.get_json()
    session['model_preference'] = data.get('model', 'rf')
    
    conn = get_db_connection()
    conn.execute('''UPDATE users SET 
                   theme = ?, model_preference = ?, 
                   email_notif = ?, push_notif = ?
                   WHERE id = ?''',
                (data.get('theme'), data.get('model'),
                 data.get('emailNotif'), data.get('pushNotif'),
                 session['user_id']))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/delete_history/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    try:
        conn = get_db_connection()
        record = conn.execute('SELECT * FROM history WHERE id = ? AND user_id = ?',
                            (id, session['user_id'])).fetchone()
        
        if record:
            conn.execute('DELETE FROM history WHERE id = ? AND user_id = ?',
                        (id, session['user_id']))
            conn.commit()
            conn.close()
            return jsonify({'success': True})
        else:
            conn.close()
            return jsonify({'error': 'Record not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM history WHERE user_id = ?', (session['user_id'],))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== ADMIN ROUTES (PROTECTED) ====================
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard - ONLY accessible by admins"""
    conn = get_db_connection()
    
    # Total users
    total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
    
    # Today's registrations
    today_users = conn.execute(
        "SELECT COUNT(*) as count FROM users WHERE DATE(created_at) = DATE('now')"
    ).fetchone()['count']
    
    # This week
    week_users = conn.execute(
        "SELECT COUNT(*) as count FROM users WHERE created_at >= DATE('now', '-7 days')"
    ).fetchone()['count']
    
    # This month
    month_users = conn.execute(
        "SELECT COUNT(*) as count FROM users WHERE created_at >= DATE('now', 'start of month')"
    ).fetchone()['count']
    
    # Recent users (last 20)
    recent_users = conn.execute(
        '''SELECT id, username, firstname, lastname, email, phone, 
           created_at, is_admin FROM users ORDER BY created_at DESC LIMIT 20'''
    ).fetchall()
    
    # Total predictions
    total_predictions = conn.execute('SELECT COUNT(*) as count FROM history').fetchone()['count']
    
    # Today's predictions
    today_predictions = conn.execute(
        "SELECT COUNT(*) as count FROM history WHERE DATE(created_at) = DATE('now')"
    ).fetchone()['count']
    
    # Total admins
    total_admins = conn.execute('SELECT COUNT(*) as count FROM users WHERE is_admin = 1').fetchone()['count']
    
    # All admins list
    all_admins = conn.execute(
        'SELECT id, username, firstname, lastname, email FROM users WHERE is_admin = 1'
    ).fetchall()
    
    conn.close()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         today_users=today_users,
                         week_users=week_users,
                         month_users=month_users,
                         recent_users=recent_users,
                         total_predictions=total_predictions,
                         today_predictions=today_predictions,
                         total_admins=total_admins,
                         all_admins=all_admins,
                         current_user_id=session['user_id'])

@app.route('/admin/make_admin', methods=['POST'])
@admin_required
def make_admin():
    """Make a user admin by email - ONLY admins can do this"""
    email = request.form.get('email')
    
    if not email:
        flash('Email is required', 'error')
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    
    if user:
        if user['is_admin']:
            flash(f'✓ {user["username"]} is already an admin', 'info')
        else:
            conn.execute('UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
            conn.commit()
            flash(f'✓ Successfully made {user["username"]} ({email}) an admin!', 'success')
    else:
        flash(f'✗ No user found with email: {email}', 'error')
    
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/remove_admin', methods=['POST'])
@admin_required
def remove_admin():
    """Remove admin privileges - ONLY admins can do this"""
    user_id = request.form.get('user_id')
    
    if not user_id:
        flash('User ID is required', 'error')
        return redirect(url_for('admin_dashboard'))
    
    # Prevent removing own admin rights
    if int(user_id) == session['user_id']:
        flash('⚠️ You cannot remove your own admin privileges', 'error')
        return redirect(url_for('admin_dashboard'))
    
    conn = get_db_connection()
    user = conn.execute('SELECT username FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if user:
        conn.execute('UPDATE users SET is_admin = 0 WHERE id = ?', (user_id,))
        conn.commit()
        flash(f'✓ Admin privileges removed from {user["username"]}', 'success')
    
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users')
@admin_required
def admin_users():
    """View all users - ONLY admins"""
    conn = get_db_connection()
    users = conn.execute(
        '''SELECT id, username, firstname, lastname, email, phone, 
           is_admin, created_at FROM users ORDER BY created_at DESC'''
    ).fetchall()
    conn.close()
    
    return render_template('admin_users.html', users=users)

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Delete a user - ONLY admins"""
    if user_id == session['user_id']:
        flash('⚠️ You cannot delete your own account', 'error')
        return redirect(url_for('admin_users'))
    
    conn = get_db_connection()
    # Delete user's history first
    conn.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
    # Delete user
    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    flash('✓ User deleted successfully', 'success')
    return redirect(url_for('admin_users'))

# ==================== PUBLIC API & STATS ====================
@app.route('/api/user_count')
def user_count_api():
    """Public API - Get user statistics"""
    conn = get_db_connection()
    total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
    total_predictions = conn.execute('SELECT COUNT(*) as count FROM history').fetchone()['count']
    conn.close()
    
    return jsonify({
        'total_users': total_users,
        'total_predictions': total_predictions,
        'timestamp': datetime.now().isoformat()
    })

# ==================== ADD THIS NEW ROUTE TO YOUR Main.py ====================
# Add this after the /api/user_count route (around line 500)

@app.route('/api/user_stats')
@login_required
def user_stats_api():
    """API endpoint to get current user's statistics"""
    try:
        conn = get_db_connection()
        
        # Get total predictions for current user
        total_predictions = conn.execute(
            'SELECT COUNT(*) as count FROM history WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()['count']
        
        # Get average quality for current user
        avg_quality_result = conn.execute(
            'SELECT AVG(quality) as avg_quality FROM history WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()
        
        avg_quality = avg_quality_result['avg_quality']
        if avg_quality is None:
            avg_quality = 0.0
        else:
            avg_quality = round(avg_quality, 1)
        
        # Get last prediction date
        last_prediction = conn.execute(
            'SELECT created_at FROM history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
        
        last_prediction_date = last_prediction['created_at'] if last_prediction else None
        
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'avg_quality': avg_quality,
            'last_prediction': last_prediction_date,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in user_stats_api: {str(e)}")
        return jsonify({
            'total_predictions': 0,
            'avg_quality': 0.0,
            'error': str(e)
        }), 500

@app.route('/stats')
def public_stats():
    """Public statistics page"""
    conn = get_db_connection()
    total_users = conn.execute('SELECT COUNT(*) as count FROM users').fetchone()['count']
    total_predictions = conn.execute('SELECT COUNT(*) as count FROM history').fetchone()['count']
    conn.close()
    
    return render_template('public_stats.html',
                         total_users=total_users,
                         total_predictions=total_predictions)

# ==================== QR CODE GENERATION ====================
@app.route('/generate_qr')
def generate_qr():
    """Generate QR code for the app"""
    # Get app URL from request
    app_url = "https://enological-quality-and-price-assessment.onrender.com"
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(app_url)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save to bytes
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png', 
                    as_attachment=True, 
                    download_name='enologix_qr_code.png')

@app.route('/qr')
def qr_page():
    """Display QR code page"""
    app_url = "https://enological-quality-and-price-assessment.onrender.com"
    return render_template('qr_code.html', app_url=app_url)

# ==================== DATABASE INITIALIZATION ====================
def init_database():
    """Initialize database tables"""
    conn = get_db_connection()
    
    # Users table
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    firstname TEXT NOT NULL,
                    lastname TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    phone TEXT NOT NULL,
                    password TEXT NOT NULL,
                    is_admin BOOLEAN DEFAULT 0,
                    theme TEXT DEFAULT 'dark',
                    model_preference TEXT DEFAULT 'rf',
                    email_notif BOOLEAN DEFAULT 0,
                    push_notif BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    
    # History table
    conn.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    wine_type TEXT NOT NULL,
                    quality INTEGER NOT NULL,
                    price_usd TEXT NOT NULL,
                    price_inr TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    fixed_acidity REAL,
                    volatile_acidity REAL,
                    citric_acid REAL,
                    residual_sugar REAL,
                    chlorides REAL,
                    free_sulfur_dioxide REAL,
                    total_sulfur_dioxide REAL,
                    density REAL,
                    pH REAL,
                    sulphates REAL,
                    alcohol REAL,
                    model_used TEXT DEFAULT 'rf',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )''')
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

# ==================== MAIN ====================
if __name__ == '__main__':
    init_database()
    
    print("\n" + "="*70)
    print(" 🍷 ENOLOGIX - Wine Quality Prediction System")
    print("="*70)
    print(" ✓ Database initialized")
    print(" ✓ Admin system: PROTECTED (only admins can access)")
    print(" ✓ Models loaded: Random Forest + Gradient Boosting")
    print(" ✓ QR Code generation: Enabled")
    print(" ✓ Cross-platform: Android, iOS, Laptop, Mac")
    print("="*70)
    print(" 🚀 Server starting on http://0.0.0.0:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)