from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(pages)

def setup_knowledge_base(docs):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)

st.title("PDF Chatbot 📄")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    docs = process_pdf("temp.pdf")
    db = setup_knowledge_base(docs)
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    if prompt := st.chat_input("Ask about the PDF"):
        relevant_docs = db.similarity_search(prompt, k=3)
        context = "\n\n".join([d.page_content for d in relevant_docs])
        response = llm.invoke(f"Context: {context}\n\nQuestion: {prompt}\nAnswer:")
        st.write(response.content)
