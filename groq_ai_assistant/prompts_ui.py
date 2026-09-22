import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import groq

# Import database & auth functions
from database import (
    sign_up_user,
    sign_in_user,
    sign_out_user,
    create_new_chat,
    get_all_chats,
    get_chat_messages,
    save_message,
    update_chat_title,
    delete_chat
)

load_dotenv()

st.set_page_config(page_title="Universal AI Assistant", page_icon="🤖", layout="wide")

# Helper to format LaTeX math formulas safely
def fix_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\w)\[\s*(\\[a-zA-Z]+.*?)\s*\]', r'$$\1$$', text, flags=re.DOTALL)
    return text

def summarize_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    try:
        summary_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2)
        summary = summary_llm.invoke(f"Summarize the key takeaways of this AI response in 2-3 short sentences:\n\n{text}")
        return f"[Summary of previous response]: {summary.content}"
    except Exception:
        return text[:max_chars] + "... [truncated for context size]"

# ==========================================
# AUTHENTICATION GUARD
# ==========================================
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Universal AI Assistant")
        st.subheader("Please sign in or create an account to continue")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                login_email = st.text_input("Email")
                login_password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Sign In", type="primary", use_container_width=True)

                if login_btn:
                    try:
                        res = sign_in_user(login_email, login_password)
                        st.session_state.user = res.user
                        st.success("Logged in successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")

        with tab2:
            with st.form("signup_form"):
                signup_email = st.text_input("Email")
                signup_password = st.text_input("Password", type="password")
                signup_btn = st.form_submit_button("Create Account", type="primary", use_container_width=True)

                if signup_btn:
                    try:
                        res = sign_up_user(signup_email, signup_password)
                        st.session_state.user = res.user
                        st.success("Account created! Logging you in...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")

    st.stop()  # Halt execution until authenticated

# ==========================================
# AUTHENTICATED USER SESSION INITIALIZATION
# ==========================================
user_id = st.session_state.user.id
user_email = st.session_state.user.email

all_chats = get_all_chats(user_id)

if "active_chat_id" not in st.session_state:
    if all_chats:
        st.session_state.active_chat_id = all_chats[0]["chat_id"]
    else:
        st.session_state.active_chat_id = create_new_chat(user_id)

# Verify active chat belongs to current list or reset
if all_chats and st.session_state.active_chat_id not in [c["chat_id"] for c in all_chats]:
    st.session_state.active_chat_id = all_chats[0]["chat_id"]

# ==========================================
# SIDEBAR: CHAT THREADS & USER PROFILE
# ==========================================
with st.sidebar:
    st.write(f"👤 **{user_email}**")
    if st.button("🚪 Sign Out", use_container_width=True):
        sign_out_user()
        st.session_state.user = None
        if "active_chat_id" in st.session_state:
            del st.session_state.active_chat_id
        st.rerun()

    st.markdown("---")
    st.title("🤖 Chat Sessions")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = create_new_chat(user_id)
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("---")
    st.subheader("Past Conversations")

    all_chats = get_all_chats(user_id)
    for chat in all_chats:
        c_id = chat["chat_id"]
        c_title = chat["title"]

        col1, col2 = st.sidebar.columns([0.8, 0.2])
        is_active = (c_id == st.session_state.active_chat_id)

        with col1:
            if st.button(f"💬 {c_title}", key=f"btn_{c_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.active_chat_id = c_id
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{c_id}"):
                delete_chat(c_id)
                if st.session_state.active_chat_id == c_id:
                    remaining_chats = get_all_chats(user_id)
                    if remaining_chats:
                        st.session_state.active_chat_id = remaining_chats[0]["chat_id"]
                    else:
                        st.session_state.active_chat_id = create_new_chat(user_id)
                st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.title("Universal AI Assistant")

current_messages = get_chat_messages(st.session_state.active_chat_id)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7, streaming=True)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert AI assistant.\n\n"
        "STRICT LATEX MATH RULES:\n"
        "- ALWAYS enclose display/block equations using double dollar signs: $$...$$\n"
        "- ALWAYS enclose inline math using single dollar signs: $...$\n"
        "- NEVER use square brackets [ ... ] or parenthetical syntax ( ... ) for math equations.\n\n"
        "FORMATTING RULES:\n"
        "- Wrap code in fenced code blocks with language tags (e.g., ```python).\n"
        "- Use standard Markdown tables, bold headers, and bullet points."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Render chat messages or empty state
if not current_messages:
    st.markdown("### 👋 How can I help you today?")
    st.caption("Ask a question, analyze code, or pick a sample topic to get started.")

    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Analyze & Explain**\n\nExplain LLM architectures, transformers, or complex algorithms step-by-step.")
        st.info("💻 **Code & Debug**\n\nWrite, debug, and optimize Python, JavaScript, SQL, or React code.")
    with col2:
        st.info("📝 **Summarize & Draft**\n\nSummarize long articles, draft professional reports, or outline ideas.")
        st.info("🧮 **Math & Science**\n\nSolve equations and formulas with step-by-step LaTeX formatting.")
else:
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(fix_latex(msg["content"]))

# Handle input
if user_input := st.chat_input("Type your message..."):
    if len(current_messages) == 0:
        update_chat_title(st.session_state.active_chat_id, user_input)

    save_message(st.session_state.active_chat_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    recent_history = current_messages[-5:]
    history_tuples = []

    for m in recent_history:
        role = "human" if m["role"] == "user" else "ai"
        content = m["content"]

        if role == "ai":
            content = summarize_text(content, max_chars=1200)
        elif role == "human" and len(content) > 2000:
            content = content[:2000] + "... [truncated]"

        history_tuples.append((role, content))

    with st.chat_message("assistant"):
        try:
            response_stream = chain.stream({
                "history": history_tuples,
                "input": user_input
            })
            full_response = st.write_stream(response_stream)
            cleaned_response = fix_latex(full_response)

            save_message(st.session_state.active_chat_id, "assistant", cleaned_response)
            st.rerun()

        except groq.APIError as e:
            st.error(f"API Error: {e}")