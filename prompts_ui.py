import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import groq

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Universal AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Universal AI Assistant")

# --- Sidebar: Reset Memory ---
with st.sidebar:
    st.header("Settings")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
        st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# --- Model & Chain Setup ---
llm = ChatGroq(
    model="groq/compound",
    temperature=0.7,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert AI assistant.\n\n"
        "STRICT LATEX MATH RULES:\n"
        "- ALWAYS enclose display/block equations using double dollar signs: $$...$$\n"
        "- ALWAYS enclose inline math using single dollar signs: $...$\n"
        "- NEVER use square brackets [ ... ], parenthetical syntax ( ... ), \\[... \\], or \\(... \\) for math equations.\n\n"
        "FORMATTING RULES:\n"
        "- Wrap code in fenced code blocks with language tags (e.g., ```python).\n"
        "- Use standard Markdown tables, bold headers, bullet points, and appropriate emojis."
    )),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# --- Render Existing Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle New User Input ---
if user_input := st.chat_input("Ask a question, request code, or solve an equation..."):
    # 1. Store and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Limit history to the LAST 10 MESSAGES to avoid 'Request Entity Too Large'
    MAX_HISTORY = 10
    recent_messages = st.session_state.messages[:-1][-MAX_HISTORY:]
    history_tuples = [(m["role"], m["content"]) for m in recent_messages]

    # 3. Stream model response safely
    with st.chat_message("assistant"):
        try:
            response_stream = chain.stream({
                "history": history_tuples,
                "input": user_input
            })
            full_response = st.write_stream(response_stream)
            
            # 4. Save model response to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except groq.APIError as e:
            if "Request Entity Too Large" in str(e) or e.status_code == 413:
                st.error("⚠️ Conversation history became too large. Clearing older context...")
                # Automatically trim history on payload overflow
                st.session_state.messages = st.session_state.messages[-4:]
            else:
                st.error(f"An API error occurred: {e}")



