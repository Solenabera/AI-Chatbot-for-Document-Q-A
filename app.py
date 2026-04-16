import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from backend.qa_pipeline import build_vectorstore, build_qa_chain

# =========================
# Load environment variables
# =========================
load_dotenv()

# Optional but good practice
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =========================
# Streamlit UI config
# =========================
st.set_page_config(page_title="AI Document Chatbot", layout="wide")

st.title("📄 AI Document Chatbot")
st.caption("Ask questions from your PDF using AI")

# =========================
# Session state (chat memory)
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# =========================
# Upload PDF
# =========================
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        vectorstore = build_vectorstore(file_path)
        st.session_state.qa_chain = build_qa_chain(vectorstore)

    st.success("Document ready!")

# =========================
# Chat input
# =========================
query = st.text_input("Ask your question")

if query and st.session_state.qa_chain:

    result = st.session_state.qa_chain(query)

    answer = result["result"]
    sources = result.get("source_documents", [])

    # Save chat history
    st.session_state.chat_history.append((query, answer))

    # =========================
    # Display chat history
    # =========================
    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.divider()

    # =========================
    # Show sources
    # =========================
    with st.expander("📚 Sources"):
        for doc in sources:
            st.write(doc.page_content[:300])