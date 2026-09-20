import os
import uuid
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Load environment variables from local .env
load_dotenv()

# 2. Safe secret retrieval helper (Local .env vs Streamlit Cloud secrets)
def get_secret(key: str) -> str | None:
    """Safely retrieves secrets from local environment variables or Streamlit Cloud."""
    val = os.getenv(key)
    if val:
        return val.strip()
    try:
        if key in st.secrets:
            return st.secrets[key].strip()
    except Exception:
        pass
    return None

# 3. Retrieve credentials
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")

# 4. Validate credentials before initializing client
if not SUPABASE_URL or not SUPABASE_URL.startswith("https://"):
    st.error(f"❌ Invalid or missing SUPABASE_URL: '{SUPABASE_URL}'")
    st.info("Check your .env or .streamlit/secrets.toml file and ensure SUPABASE_URL starts with 'https://'")
    st.stop()

if not SUPABASE_KEY:
    st.error("❌ Missing SUPABASE_KEY in .env or secrets.toml")
    st.stop()

# 5. Initialize Supabase client with Streamlit caching
@st.cache_resource
def init_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# ==========================================
# DATABASE CRUD FUNCTIONS
# ==========================================

def create_new_chat() -> str:
    chat_id = str(uuid.uuid4())
    supabase.table("chats").insert({
        "chat_id": chat_id,
        "title": "New Chat"
    }).execute()
    return chat_id

def get_all_chats():
    response = supabase.table("chats").select("chat_id, title").order("created_at", desc=True).execute()
    return response.data

def get_chat_messages(chat_id: str):
    response = supabase.table("messages").select("role, content, summary").eq("chat_id", chat_id).order("id", desc=False).execute()
    return response.data

def save_message(chat_id: str, role: str, content: str, summary: str = None):
    supabase.table("messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "summary": summary
    }).execute()

def update_chat_title(chat_id: str, first_message: str):
    title = first_message[:30] + ("..." if len(first_message) > 30 else "")
    supabase.table("chats").update({"title": title}).eq("chat_id", chat_id).execute()

def delete_chat(chat_id: str):
    supabase.table("chats").delete().eq("chat_id", chat_id).execute()