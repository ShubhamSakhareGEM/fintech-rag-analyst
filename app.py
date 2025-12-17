__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

st.set_page_config(page_title="FinSight: 10-K Analyzer", layout="wide")

with st.sidebar:
    st.title("Settings")
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("API Key loaded from Secrets")
    else:
        api_key = st.text_input("Enter Google API Key:", type="password")
        
    uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
    process_button = st.button("Process Documents")

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for the LLM."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Generates embeddings and stores them in FAISS using a local model."""

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    """Creates the QA chain using the updated Gemini 2.5 model."""
    prompt_template = """
    You are a financial analyst assistant. Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    """Handles the user query."""
    try:

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("### Analysis:")
        st.write(response["output_text"])
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("""
## 📊 FinSight: Financial Document Analyst
*Upload a financial PDF (e.g., 10-K report) and ask questions.*
""")

if process_button and uploaded_files and api_key:
    with st.spinner("Processing... This may take a minute for large files."):
        raw_text = get_pdf_text(uploaded_files)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done! You can now ask questions.")
        else:
            st.error("Could not extract text from the PDF. It might be scanned/image-based.")

user_question = st.text_input("Ask a question about the report:")

if user_question:
    if not api_key:
        st.warning("Please enter your API Key in the sidebar first.")
    elif not os.path.exists("faiss_index"):
        st.warning("Please upload and process a PDF file first.")
    else:
        user_input(user_question, api_key)
