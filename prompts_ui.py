import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="Universal AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Universal AI Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]


llm = ChatGroq(
    model="groq/compound",
    temperature=0.7,
    streaming=True  # Enables real-time streaming
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

parser = StrOutputParser()

chain = prompt | llm | parser


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input := st.chat_input("Ask a question, request code, or solve an equation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    
    history_tuples = []
    for m in st.session_state.messages[:-1]:  # Exclude current query
        history_tuples.append((m["role"], m["content"]))

    
    with st.chat_message("assistant"):
        # st.write_stream renders tokens live and automatically parses Markdown, Math, and Code
        response_stream = chain.stream({
            "history": history_tuples,
            "input": user_input
        })
        full_response = st.write_stream(response_stream)

    
    st.session_state.messages.append({"role": "assistant", "content": full_response})



