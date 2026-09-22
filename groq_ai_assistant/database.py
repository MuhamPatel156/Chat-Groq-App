import os
import uuid
import hashlib
import secrets
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Helper to safely retrieve environment variables or st.secrets
def get_secret(key: str) -> str | None:
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        if key in st.secrets:
            return st.secrets[key].strip()
    except Exception:
        pass
    return None

DATABASE_URL = get_secret("DATABASE_URL")

if not DATABASE_URL:
    st.error("❌ Missing DATABASE_URL in environment variables or st.secrets")
    st.stop()

def get_connection():
    return psycopg2.connect(DATABASE_URL)

# Simple wrapper objects to maintain compatibility with prompts_ui.py
class User:
    def __init__(self, user_id: str, email: str):
        self.id = user_id
        self.email = email

class AuthResponse:
    def __init__(self, user: User):
        self.user = user

# ==========================================
# PASSWORD SECURITY HELPERS
# ==========================================

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt

def verify_password(stored_hash: str, salt: str, password: str) -> bool:
    new_hash, _ = hash_password(password, salt)
    return new_hash == stored_hash

# ==========================================
# AUTHENTICATION FUNCTIONS
# ==========================================

def sign_up_user(email: str, password: str) -> AuthResponse:
    if not email or not password:
        raise ValueError("Email and password are required.")

    pwd_hash, salt = hash_password(password)

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check if user exists
            cur.execute("SELECT id FROM users WHERE email = %s;", (email.lower().strip(),))
            if cur.fetchone():
                raise Exception("An account with this email already exists.")

            # Insert new user
            cur.execute(
                "INSERT INTO users (email, password_hash, salt) VALUES (%s, %s, %s) RETURNING id, email;",
                (email.lower().strip(), pwd_hash, salt)
            )
            user_data = cur.fetchone()
            conn.commit()

    return AuthResponse(User(str(user_data["id"]), user_data["email"]))

def sign_in_user(email: str, password: str) -> AuthResponse:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, password_hash, salt FROM users WHERE email = %s;",
                (email.lower().strip(),)
            )
            user_data = cur.fetchone()

            if not user_data or not verify_password(user_data["password_hash"], user_data["salt"], password):
                raise Exception("Invalid email or password.")

    return AuthResponse(User(str(user_data["id"]), user_data["email"]))

def sign_out_user():
    pass  # Session cleanup is handled in Streamlit session state

# ==========================================
# CHAT DATABASE CRUD FUNCTIONS
# ==========================================

def create_new_chat(user_id: str) -> str:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO chats (user_id, title) VALUES (%s, %s) RETURNING chat_id;",
                (user_id, "New Chat")
            )
            chat_id = str(cur.fetchone()["chat_id"])
            conn.commit()
    return chat_id

def get_all_chats(user_id: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT chat_id::text, title FROM chats WHERE user_id = %s ORDER BY created_at DESC;",
                (user_id,)
            )
            return cur.fetchall()

def get_chat_messages(chat_id: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, summary FROM messages WHERE chat_id = %s ORDER BY id ASC;",
                (chat_id,)
            )
            return cur.fetchall()

def save_message(chat_id: str, role: str, content: str, summary: str = None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (chat_id, role, content, summary) VALUES (%s, %s, %s, %s);",
                (chat_id, role, content, summary)
            )
            conn.commit()

def update_chat_title(chat_id: str, first_message: str):
    title = first_message[:30] + ("..." if len(first_message) > 30 else "")
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chats SET title = %s WHERE chat_id = %s;",
                (title, chat_id)
            )
            conn.commit()

def delete_chat(chat_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chats WHERE chat_id = %s;", (chat_id,))
            conn.commit()