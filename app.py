import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated for cloud compatibility

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide")
st.title("🛡️ GxP-Validated AI Knowledge Assistant..by Shan.R")
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
    path = "knowledge/"
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.error("Missing PDF in /knowledge folder!")
        return None
    
    all_docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(path, pdf))
        all_docs.extend(loader.load()) # This combines all PDFs into one "brain"
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)
    return vectorstore

# --- RUNNING THE APP ---
engine = setup_engine()

# Initialize Groq inside a cached function so it's only done once
def get_llm():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        temperature=0
    )

user_input = st.text_input("Ask a question about your SOP:")

if user_input and engine:
    # 1. Log the activity for Audit Trail
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": now, "text": user_input})
    
    # 2. Search & Answer
    with st.spinner("Consulting validated documents..."):
        # Search for context
        results = engine.similarity_search(user_input, k=3)
        context = results[0].page_content
        
        # Connect to Groq
        llm = get_llm()
        
        # Generate Response
        prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
        response = llm.invoke(prompt)
        
        # 3. Display Results
        st.write("### AI Response:")
        st.success(response.content)
        st.info(f"📄 Source: {results[0].metadata['source']}")



