from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_this_in_production'

# Load both models
try:
    # Random Forest Model
    with open('wine_quality_rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Gradient Boosting Model (if available)
    try:
        with open('wine_quality_gb_model.pkl', 'rb') as f:
            gb_model = pickle.load(f)
    except FileNotFoundError:
        gb_model = rf_model  # Fallback to RF if GB not available
    
    # Scaler and features
    with open('wine_quality_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('wine_quality_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("✓ Models loaded successfully!")
    print(f"✓ Random Forest: Available")
    print(f"✓ Gradient Boosting: {'Available' if gb_model != rf_model else 'Using RF fallback'}")
    
except FileNotFoundError as e:
    print(f"ERROR: Model files not found: {e.filename}")
    rf_model = None
    gb_model = None
    scaler = None
    feature_names = None

# Database connection
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create features from input
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
    
    # Engineered features
    features['acidity_ratio'] = fixed_acidity / (volatile_acidity + 0.001)
    features['sulfur_ratio'] = free_sulfur_dioxide / (total_sulfur_dioxide + 0.001)
    features['acid_sugar_interaction'] = citric_acid * residual_sugar
    features['alcohol_density'] = alcohol / density
    features['sulphates_alcohol'] = sulphates * alcohol
    features['total_acidity'] = fixed_acidity + volatile_acidity + citric_acid
    features['bound_sulfur'] = total_sulfur_dioxide - free_sulfur_dioxide
    
    return features

# Predict wine type
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

# Calculate price
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

# Routes
@app.route('/')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

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
            flash('Username already exists!', 'error')
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
            flash(f'Welcome back, {user["firstname"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def result():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if rf_model is None:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('prediction'))

    try:
        # Get input values
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

        # Create features
        features_dict = create_features(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        )

        # Prepare input
        input_df = pd.DataFrame([features_dict])
        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        # Get model preference from session or use default
        model_choice = session.get('model_preference', 'rf')
        model = rf_model if model_choice == 'rf' else gb_model

        # Predict quality
        predicted_quality = int(model.predict(input_scaled)[0])
        
        # Predict wine type
        wine_type, wine_confidence = predict_wine_type(
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
            density, pH, sulphates, alcohol
        )

        # Calculate price
        price_usd = calculate_wine_price(
            predicted_quality, alcohol, volatile_acidity, residual_sugar,
            pH, sulphates, wine_type
        )
        price_inr = price_usd * 83

        # Save to history
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
def performance():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('performance.html')

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        conn = get_db_connection()
        history_data = conn.execute(
            'SELECT * FROM history WHERE user_id = ? ORDER BY created_at DESC',
            (session['user_id'],)
        ).fetchall()
        conn.close()
        
        # Convert Row objects to dictionaries for JSON serialization
        history_list = []
        for row in history_data:
            history_list.append({
                'id': row['id'],
                'wine_type': row['wine_type'],
                'quality': row['quality'],
                'price_usd': row['price_usd'],
                'price_inr': row['price_inr'],
                'confidence': row['confidence'],
                'fixed_acidity': row['fixed_acidity'],
                'volatile_acidity': row['volatile_acidity'],
                'citric_acid': row['citric_acid'],
                'residual_sugar': row['residual_sugar'],
                'chlorides': row['chlorides'],
                'free_sulfur_dioxide': row['free_sulfur_dioxide'],
                'total_sulfur_dioxide': row['total_sulfur_dioxide'],
                'density': row['density'],
                'pH': row['pH'],
                'sulphates': row['sulphates'],
                'alcohol': row['alcohol'],
                'model_used': row['model_used'],
                'created_at': row['created_at']
            })
        
        return render_template('history.html', history=history_list)
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        flash('Error loading history. Please try again.', 'error')
        return render_template('history.html', history=[])

@app.route('/settings')
def settings():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('settings.html')

@app.route('/save_settings', methods=['POST'])
def save_settings():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    session['model_preference'] = data.get('model', 'rf')
    
    # Save to database
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
def delete_history(id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        # Verify the record belongs to the user
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
        print(f"Error deleting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM history WHERE user_id = ?', (session['user_id'],))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

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
import qrcode
import io
from flask import send_file

# -------------------- QR CODE ROUTES --------------------

@app.route('/generate_qr')
def generate_qr():
    """Generate QR code for the app"""
    app_url = request.host_url  # Example: http://127.0.0.1:5000/
    
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
    
    # Save to BytesIO
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    # Send image file directly
    return send_file(img_io, mimetype='image/png',
                     as_attachment=True,
                     download_name='enologix_qr_code.png')


@app.route('/qr')
def qr_page():
    """Display QR code page"""
    app_url = request.host_url
    return render_template('qr_code.html', app_url=app_url)


if __name__ == '__main__':
    # Create database tables
    conn = get_db_connection()
    
    # Users table
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    firstname TEXT NOT NULL,
                    lastname TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    password TEXT NOT NULL,
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
    
    print("\n" + "="*60)
    print("ENOLOGIX - Wine Quality Prediction System")
    print("="*60)
    print("✓ Database initialized")
    print("✓ Models loaded (RF + GB)")
    print("✓ History tracking enabled")
    print("✓ Theme support enabled")
    print("✓ Server starting...")
    print("="*60 + "\n")

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)