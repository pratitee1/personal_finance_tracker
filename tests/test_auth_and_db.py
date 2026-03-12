from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.dependencies import auth as auth_deps
from api.routes import auth as auth_routes
from db.models import Base
from db.models.user import User
from db import setup as db_setup


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_register_login_and_me_flow(db_session, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")

    registered_user = auth_routes.register_user(
        auth_routes.UserCreate(
            email="user@example.com",
            name="Test User",
            password="password123",
        ),
        db_session,
    )

    assert registered_user.email == "user@example.com"
    assert registered_user.name == "Test User"
    assert registered_user.oauth_provider is None

    stored_user = db_session.query(User).filter(User.email == "user@example.com").first()
    assert stored_user is not None
    assert stored_user.hashed_password != "password123"
    assert auth_deps.verify_password("password123", stored_user.hashed_password)

    token_payload = auth_routes.login_for_access_token(
        SimpleNamespace(username="user@example.com", password="password123"),
        db_session,
    )

    assert token_payload["token_type"] == "bearer"
    assert token_payload["access_token"]

    current_user = auth_deps.get_current_user(
        token=token_payload["access_token"],
        db=db_session,
    )

    me_response = auth_routes.read_current_user(current_user)
    assert me_response.email == "user@example.com"


def test_register_duplicate_email_returns_conflict(db_session):
    payload = {
        "email": "duplicate@example.com",
        "name": "Dup User",
        "password": "password123",
    }

    first = auth_routes.register_user(auth_routes.UserCreate(**payload), db_session)
    assert first.email == "duplicate@example.com"

    with pytest.raises(HTTPException) as exc_info:
        auth_routes.register_user(auth_routes.UserCreate(**payload), db_session)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Email is already registered"


def test_login_rejects_incorrect_password(db_session):
    auth_routes.register_user(
        auth_routes.UserCreate(
            email="wrongpass@example.com",
            name="Wrong Pass",
            password="password123",
        ),
        db_session,
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_routes.login_for_access_token(
            SimpleNamespace(username="wrongpass@example.com", password="bad-password"),
            db_session,
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Incorrect email or password"


def test_decode_access_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")

    with pytest.raises(HTTPException) as exc_info:
        auth_deps.decode_access_token("not-a-real-token")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Could not validate credentials"


def test_create_access_token_uses_custom_expiry(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")

    token = auth_deps.create_access_token(
        subject="expiry@example.com",
        expires_delta=timedelta(minutes=5),
    )

    payload = auth_deps.decode_access_token(token)
    assert payload["sub"] == "expiry@example.com"
    assert "exp" in payload


def test_get_current_user_handles_missing_and_inactive_users(db_session, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")

    missing_user_token = auth_deps.create_access_token("missing@example.com")
    with pytest.raises(HTTPException) as missing_exc:
        auth_deps.get_current_user(token=missing_user_token, db=db_session)
    assert missing_exc.value.status_code == 401
    assert missing_exc.value.detail == "User not found"

    inactive_user = User(
        email="inactive@example.com",
        name="Inactive User",
        hashed_password=auth_deps.get_password_hash("password123"),
        is_active=False,
    )
    db_session.add(inactive_user)
    db_session.commit()

    inactive_token = auth_deps.create_access_token("inactive@example.com")
    with pytest.raises(HTTPException) as inactive_exc:
        auth_deps.get_current_user(token=inactive_token, db=db_session)
    assert inactive_exc.value.status_code == 400
    assert inactive_exc.value.detail == "Inactive user"


def test_authenticate_user_returns_none_for_missing_hash(db_session):
    user = User(email="oauth@example.com", name="OAuth Only", hashed_password=None)
    db_session.add(user)
    db_session.commit()

    assert auth_deps.authenticate_user(db_session, "oauth@example.com", "password123") is None


def test_get_db_yields_session_and_closes_it(monkeypatch):
    fake_session = MagicMock()
    monkeypatch.setattr(db_setup, "SessionLocal", lambda: fake_session)

    session_generator = db_setup.get_db()
    yielded_session = next(session_generator)

    assert yielded_session is fake_session

    with pytest.raises(StopIteration):
        next(session_generator)

    fake_session.close.assert_called_once()
