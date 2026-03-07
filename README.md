<div align="center">

# 🍷 Enologix  
### Wine Quality & Price Predictor

<p>
  🔗 <strong>Live Demo:</strong><br>
  <a href="https://enologix.onrender.com" target="_blank">
    https://enologix.onrender.com
  </a>
</p>

<br>

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white"/>
</p>

<br>

<p>
  A modern full-stack web application that leverages <strong>Machine Learning</strong>  
  to predict <strong>wine quality (0–10)</strong> and estimate  
  <strong>real-world price (USD & INR)</strong> using physicochemical properties.
</p>

<p>
  Built with authentication, history tracking, admin controls,  
  and a clean, fully responsive user interface.
</p>

</div>

---

# ✨ Features

- 🔐 Secure user registration & login (hashed passwords)  
- ⚡ Instant wine quality & price prediction using **Random Forest** (with Gradient Boosting fallback)  
- 🧠 Engineered features + StandardScaler for high accuracy  
- 🍷 Automatic red/white wine classification with confidence %  
- 📜 Full prediction history (view / delete / clear)  
- 👑 Admin dashboard (user stats, make/remove admin)  
- ⚙️ User settings: dark/light theme, preferred model, notifications  
- 📱 QR code generator for quick mobile access  
- 📲 Fully responsive & mobile-friendly design  
- ☁️ Cloud database with **MongoDB Atlas** (persistent & secure)  
- 🌍 Deployed on **Render.com** with free HTTPS  

---

# 🛠️ Tech Stack

| Category           | Technology                              |
|--------------------|------------------------------------------|
| **Backend**        | Flask (Python)                          |
| **Database**       | MongoDB Atlas (Cloud)                   |
| **Machine Learning** | scikit-learn, pandas, numpy          |
| **Authentication** | Werkzeug, Flask-Session                 |
| **Frontend**       | HTML5, CSS3, Jinja2                     |
| **Deployment**     | Render.com (Free Tier)                  |
| **Environment**    | python-dotenv, PyMongo, Gunicorn        |
| **Extras**         | qrcode, Pillow                          |

---

# 📂 Project Structure

```
Wine_quality/
│
├── Main.py                     # 🚀 Core Flask application
├── requirements.txt            # 📦 Python dependencies
├── .env                        # 🔐 Environment variables (DO NOT COMMIT)
├── .gitignore
│
├── templates/                  # 🎨 Jinja2 HTML templates
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── prediction.html
│   ├── result.html
│   ├── history.html
│   ├── settings.html
│   ├── admin_dashboard.html
│   ├── public_stats.html
│   ├── qr_code.html
│   └── ...
│
├── static/                     # 🎨 CSS, JS, images
│
└── wine_quality_*.pkl          # 🤖 Trained ML models & scaler
```

---

# 🚀 Local Development Setup

Follow these steps to run the project locally:

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rahul-22-2004/Wine_quality.git
cd Wine_quality
```

---

## 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

### Windows
```bash
venv\Scripts\activate
```

### macOS / Linux
```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Create `.env` File (Important 🔐)

Create a `.env` file in the root directory and add:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/enologix?retryWrites=true&w=majority
SECRET_KEY=your-very-long-random-secret-key-here
```

⚠️ Never commit `.env` to GitHub.

---

## 5️⃣ Run the Application

```bash
python Main.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

# 🌐 Deployment (Render.com)

🔗 **Live Application:**  
https://enologix.onrender.com

### ⚙️ Deployment Configuration

**Build Command**
```bash
pip install -r requirements.txt
```

**Start Command**
```bash
gunicorn Main:app
```

**Environment Variables Required**
```
MONGO_URI
SECRET_KEY
```

**Hosting Plan**
```
Render Free Tier
Cold starts reduced using UptimeRobot (ping every 5 minutes)
```

---

# 🚀 Future Enhancements

- 👤 User profile pictures & avatars  
- 📊 Interactive charts (quality trends, analytics dashboard)  
- 📁 Export prediction history (CSV / PDF)  
- 💰 Real-time wine market price integration  
- 🧪 Unit testing with pytest  
- 🐳 Docker support  
- ⚛️ Optional React / Vue frontend upgrade  
- 📈 Model performance comparison dashboard  
