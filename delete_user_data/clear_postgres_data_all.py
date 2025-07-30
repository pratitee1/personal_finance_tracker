import os
from sqlalchemy import text
from db.setup import SessionLocal

def clear_all_data():
    db = SessionLocal()
    try:
        db.execute(text("TRUNCATE TABLE line_items RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE receipts RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE users RESTART IDENTITY CASCADE;"))
        db.commit()
        print("All PostgreSQL tables (users, receipts, line_items) have been cleared.")
    except Exception as e:
        db.rollback()
        print(f"Failed to clear tables: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_data()
