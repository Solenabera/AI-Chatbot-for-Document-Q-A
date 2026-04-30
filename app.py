import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from backend.qa_pipeline import build_vectorstore, build_qa_chain
from backend.config import DEFAULT_RETRIEVAL_K, MODEL_NAME, TEMPERATURE

# =========================
# Load environment variables
# =========================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =========================
# Streamlit UI config
# =========================
st.set_page_config(page_title="AI Document Chatbot", layout="wide")

st.title("📄 AI Document Chatbot")
st.caption("Ask questions from your PDF using AI")

# =========================
# Sidebar controls
with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("OPENAI_API_KEY")
    retrieval_k = st.slider("Document sources to retrieve", min_value=1, max_value=5, value=DEFAULT_RETRIEVAL_K)
    temperature = st.slider("Response creativity", min_value=0.0, max_value=1.0, value=float(TEMPERATURE), step=0.1)
    show_sources = st.checkbox("Show sources by default", value=True)
    if st.button("Reset conversation"):
        keys = ["chat_history", "qa_chain", "document_info", "uploaded_file_name", "vectorstore", "error_message"]
        for key in keys:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    if not api_key:
        st.error("Missing OPENAI_API_KEY. Please add it to your .env file.")

    st.markdown("---")
    st.markdown("**Current model:**")
    st.write(MODEL_NAME)
    st.markdown("**Tips:**")
    st.write("Use a precise question and upload one PDF at a time for best results.")

# =========================
# Session state (chat memory)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "document_info" not in st.session_state:
    st.session_state.document_info = {}

if "error_message" not in st.session_state:
    st.session_state.error_message = ""

# =========================
# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            vectorstore = build_vectorstore(file_path)
            st.session_state.qa_chain = build_qa_chain(
                vectorstore,
                model_name=MODEL_NAME,
                temperature=temperature,
                k=retrieval_k,
            )

            chunk_count = None
            if hasattr(vectorstore, "index") and hasattr(vectorstore.index, "ntotal"):
                chunk_count = int(vectorstore.index.ntotal)

            st.session_state.document_info = {
                "name": uploaded_file.name,
                "size": f"{uploaded_file.size / 1024:.1f} KB",
                "chunks": chunk_count or "unknown",
                "retrieval_k": retrieval_k,
                "temperature": temperature,
            }
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.error_message = ""
            st.success("Document ready!")
        except Exception as exc:
            st.session_state.qa_chain = None
            st.session_state.error_message = str(exc)
            st.error("Failed to process the PDF or connect to the OpenAI API.")
            st.exception(exc)

# =========================
# Document status and controls
if st.session_state.document_info:
    status_cols = st.columns(3)
    status_cols[0].metric("Uploaded file", st.session_state.document_info.get("name", "—"))
    status_cols[1].metric("File size", st.session_state.document_info.get("size", "—"))
    status_cols[2].metric("Chunks", st.session_state.document_info.get("chunks", "—"))

    st.info(
        f"Retrieving top {st.session_state.document_info.get('retrieval_k')} sources using {MODEL_NAME} at temperature {st.session_state.document_info.get('temperature')}"
    )

# =========================
# Chat input
query = st.text_input("Ask your question")

if query:
    if not st.session_state.qa_chain:
        st.warning("Upload a PDF document first, then ask your question.")
    else:
        with st.spinner("Fetching answer from the AI..."):
            try:
                result = st.session_state.qa_chain(query)
                answer = result.get("result") or result.get("answer") or ""
                sources = result.get("source_documents", [])

                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer,
                    "sources": sources,
                })
                st.session_state.error_message = ""
            except Exception as exc:
                st.session_state.error_message = str(exc)
                st.error("The API did not respond correctly. Please try again later.")
                st.exception(exc)

# =========================
# Display chat history
if st.session_state.chat_history:
    for item in st.session_state.chat_history:
        st.markdown(f"**You:** {item['question']}")
        st.markdown(f"**AI:** {item['answer']}")
        st.divider()

    transcript = "\n\n".join(
        [f"You: {item['question']}\nAI: {item['answer']}" for item in st.session_state.chat_history]
    )
    st.sidebar.download_button(
        label="Download chat transcript",
        data=transcript,
        file_name="ai_document_chat_transcript.txt",
        mime="text/plain",
    )

# =========================
# Show sources
if st.session_state.chat_history:
    if show_sources:
        with st.expander("📚 Sources", expanded=True):
            latest = st.session_state.chat_history[-1]
            for doc in latest.get("sources", []):
                page_info = doc.metadata.get("page") if doc.metadata else None
                source_label = f"Page {page_info}" if page_info else "Source"
                st.write(f"**{source_label}**")
                st.write(doc.page_content[:400])
                st.write("---")
    else:
        with st.expander("📚 Sources", expanded=False):
            st.write("Toggle 'Show sources by default' in the sidebar to view source snippets automatically.")

# =========================
# Error diagnostics
if st.session_state.error_message:
    st.error(f"Last error: {st.session_state.error_message}")
