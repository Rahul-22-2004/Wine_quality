import sqlite3

# Connect to your existing users.db file
conn = sqlite3.connect('users.db')

# Add the missing column if it doesn't exist
try:
    conn.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0")
    conn.commit()
    print("✅ 'is_admin' column added successfully!")
except sqlite3.OperationalError as e:
    print("⚠️ Column might already exist or another issue occurred:", e)

conn.close()

