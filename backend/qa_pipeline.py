from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

from backend.pdf_utils import load_and_split_pdf
from backend.config import TEMPERATURE

def build_vectorstore(file_path):
    docs = load_and_split_pdf(file_path)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

def build_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=TEMPERATURE)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )