from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from backend.pdf_utils import load_and_split_pdf
from backend.config import DEFAULT_RETRIEVAL_K, MODEL_NAME, TEMPERATURE

def build_vectorstore(file_path):
    docs = load_and_split_pdf(file_path)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_qa_chain(vectorstore, model_name=MODEL_NAME, temperature=TEMPERATURE, k=DEFAULT_RETRIEVAL_K):
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True
    )