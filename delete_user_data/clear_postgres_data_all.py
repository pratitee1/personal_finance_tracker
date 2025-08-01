import os
from sqlalchemy import text
from db.setup import SessionLocal
from db.models.user import User

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

def create_default_user():
    db = SessionLocal()
    try:
        default_user = User(
            id=1,
            name="Default User",
            email="default@example.com"
        )
        db.add(default_user)
        db.commit()
        db.execute(text(
            "SELECT setval(pg_get_serial_sequence('users','id'), (SELECT MAX(id) FROM users));"
        ))
        db.commit()
        print(f"Default user created.")
    except Exception as e:
        db.rollback()
        print(f"Failed to create default user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_data()
    create_default_user()
