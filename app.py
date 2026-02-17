import streamlit as st
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide")
st.title("🛡️ GxP-Validated AI Assistant")
st.markdown("---")

# --- SIDEBAR: AUDIT TRAIL (21 CFR Part 11) ---
with st.sidebar:
    st.header("📜 Audit Trail")
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    for entry in st.session_state.logs:
        st.caption(f"{entry['time']}: {entry['text']}")

# --- LOGIC: INITIALIZE THE BRAIN ---
@st.cache_resource
def setup_engine():
    # Load the PDF from your folder
    path = "knowledge/"
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("Missing PDF in /knowledge folder!")
        return None
    
    loader = PyPDFLoader(os.path.join(path, pdf_files[0]))
    data = loader.load()
    
    # Create the 'Search Index'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings)
    return vectorstore

# --- RUNNING THE APP ---
engine = setup_engine()

user_input = st.text_input("Ask a question about your SOP:")

if user_input and engine:
    # 1. Log the activity
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": now, "text": user_input})
    
    # 2. Search & Answer
    with st.spinner("Consulting validated documents..."):
        results = engine.similarity_search(user_input, k=1)
        context = results[0].page_content
        
        # Call Ollama (Llama3)
        llm = Ollama(model="llama3")
        response = llm.invoke(f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:")
        
        st.write("### AI Response:")
        st.success(response)
        st.info(f"Source: {results[0].metadata['source']}")