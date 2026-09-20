import re
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import groq

# Import database functions from database.py
from database import (
    create_new_chat,
    get_all_chats,
    get_chat_messages,
    save_message,
    update_chat_title,
    delete_chat
)

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Universal AI Assistant", page_icon="🤖", layout="wide")

# --- LATEX CLEANER HELPER ---
def fix_latex(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'(?<!\w)\[\s*(\\[a-zA-Z]+.*?)\s*\]', r'$$\1$$', text, flags=re.DOTALL)
    return text

# --- ON-THE-FLY SUMMARIZER FOR HISTORICAL MESSAGES ---
def summarize_text(text: str, max_chars: int = 1200) -> str:
    """Condenses long past outputs into 2-3 sentences to prevent HTTP 413 payload errors."""
    if len(text) <= max_chars:
        return text
    try:
        summary_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.2)
        summary = summary_llm.invoke(f"Summarize the key takeaways of this AI response in 2-3 short sentences:\n\n{text}")
        return f"[Summary of previous response]: {summary.content}"
    except Exception:
        return text[:max_chars] + "... [truncated for context size]"

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
all_chats = get_all_chats()

if "active_chat_id" not in st.session_state:
    if all_chats:
        st.session_state.active_chat_id = all_chats[0]["chat_id"]
    else:
        st.session_state.active_chat_id = create_new_chat()

# ==========================================
# SIDEBAR: CHAT THREADS & NEW CHAT
# ==========================================
with st.sidebar:
    st.title("🤖 Chat Sessions")
    
    # 1. New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = create_new_chat()
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("---")
    st.subheader("Past Conversations")

    # 2. Render List of Past Chats
    all_chats = get_all_chats()
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
                    remaining_chats = get_all_chats()
                    if remaining_chats:
                        st.session_state.active_chat_id = remaining_chats[0]["chat_id"]
                    else:
                        st.session_state.active_chat_id = create_new_chat()
                st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================
st.title("Universal AI Assistant")

# Fetch messages for current active chat
current_messages = get_chat_messages(st.session_state.active_chat_id)

# Setup LLM & Chain
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

# --- RENDER MESSAGES OR WELCOME SCREEN ---
if not current_messages:
    # Empty State Welcome Screen
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
    # Render existing messages cleanly once
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(fix_latex(msg["content"]))

# --- HANDLE USER INPUT ---
if user_input := st.chat_input("Type your message..."):
    # 1. Update title if this is the first message in this chat session
    if len(current_messages) == 0:
        update_chat_title(st.session_state.active_chat_id, user_input)

    # 2. Save user message to Supabase & render
    save_message(st.session_state.active_chat_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. Format last 5 messages with automated summarization/capping
    recent_history = current_messages[-5:]
    history_tuples = []

    for m in recent_history:
        # Convert DB roles to standard LangChain prompt roles
        role = "human" if m["role"] == "user" else "ai"
        content = m["content"]
        
        # Summarize long past AI responses to compress payload size
        if role == "ai":
            content = summarize_text(content, max_chars=1200)
        elif role == "human" and len(content) > 2000:
            content = content[:2000] + "... [truncated]"
            
        history_tuples.append((role, content))

    # 4. Stream response & save to Supabase
    with st.chat_message("assistant"):
        try:
            response_stream = chain.stream({
                "history": history_tuples,
                "input": user_input
            })
            full_response = st.write_stream(response_stream)
            cleaned_response = fix_latex(full_response)
            
            # Save unabridged assistant response to Supabase
            save_message(st.session_state.active_chat_id, "assistant", cleaned_response)
            st.rerun()

        except groq.APIError as e:
            st.error(f"API Error: {e}")