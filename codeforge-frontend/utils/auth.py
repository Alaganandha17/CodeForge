import streamlit as st
import json
import os
import hashlib
import hmac
import secrets
from typing import Optional, Dict


def _hash_password(password: str, salt: str = "") -> str:
    """Hash a password with SHA-256 and optional salt."""
    if not salt:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    )
    return f"{salt}${hashed.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    if "$" not in stored_hash:
        # Legacy plaintext comparison (for migration)
        return hmac.compare_digest(password, stored_hash)
    salt, hash_val = stored_hash.split("$", 1)
    check_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    )
    return hmac.compare_digest(check_hash.hex(), hash_val)


def get_users():
    raw = os.getenv("APP_USERS_JSON", "").strip()
    if not raw:
        # Default users - in production, set APP_USERS_JSON env var with hashed passwords
        return [
            {"u": "admin", "p": "admin123", "role": "admin"},
            {"u": "user", "p": "user123", "role": "user"}
        ]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def authenticate(username: str, password: str) -> Optional[Dict]:
    if not username or not password:
        return None

    for rec in get_users():
        if rec.get("u") == username:
            stored = rec.get("p", "")
            if _verify_password(password, stored):
                return {"username": username, "role": rec.get("role", "user")}
    return None


def is_authed() -> bool:
    return st.session_state.get("auth", {}).get("is_auth", False)


def get_current_user() -> Optional[Dict]:
    if is_authed():
        return st.session_state["auth"].get("user")
    return None


def require_auth():
    if not is_authed():
        st.switch_page("Login.py")


def logout():
    st.session_state.clear()
    st.switch_page("Login.py")
