import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Updated for cloud compatibility

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide")
st.title("🛡️ GxP-Validated AI Knowledge Assistant")
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
    if not os.path.exists(path):
        os.makedirs(path)
        
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("⚠️ No PDFs found in /knowledge folder. Please upload a file to GitHub.")
        return None
    
    loader = PyPDFLoader(os.path.join(path, pdf_files[0]))
    data = loader.load()
    
    # Using modern HuggingFace implementation
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings)
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
        results = engine.similarity_search(user_input, k=1)
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

