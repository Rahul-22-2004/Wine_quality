# app.py - Vercel entrypoint
from Main import app  # uses the Flask app defined in Main.py

if __name__ == "__main__":
    app.run(debug=True)
