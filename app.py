import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide")
st.title("🛡️ GxP-Validated AI Knowledge Assistant..by Shan.")
st.markdown("---")

# --- SIDEBAR: AUDIT TRAIL (21 CFR Part 11) ---
with st.sidebar:
    st.header("📜 Audit Trail")
    if 'logs' not in st.session_state:
        st.session_state.logs = []
    for entry in st.session_state.logs:
        st.caption(f"{entry['time']}: {entry['text']}")

# --- LOGIC: INITIALIZE THE BRAIN (Multi-Doc Version) ---
@st.cache_resource
def setup_engine():
    path = "knowledge/"
    if not os.path.exists(path):
        os.makedirs(path)
        
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    if not pdf_files:
        st.error("⚠️ No PDFs found in /knowledge folder.")
        return None
    
    all_pages = []
    
    # Loop through every PDF in the folder
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(path, pdf))
            all_pages.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {pdf}: {e}")

    # Create the search index from ALL documents
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=all_pages, embedding=embeddings)
    return vectorstore

# --- LOGIC: CONNECT TO GROQ BRAIN ---
def get_llm():
    # Accessing the secret from your Streamlit Vault
    groq_api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        temperature=0
    )

# --- RUNNING THE APP ---
engine = setup_engine()

# Capture User Input
user_input = st.text_input("Ask a question about your SOP library:")

if user_input and engine:
    # 1. Log activity for Audit Trail
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": now, "text": user_input})
    
    with st.spinner("Analyzing all SOPs..."):
        # 2. Search & Answer (k=3 pulls context from across multiple pages/docs)
        results = engine.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
        
        llm = get_llm()
        prompt = f"Context: {context}\n\nQuestion: {user_input}\n\nAnswer:"
        response = llm.invoke(prompt)
        
        # 3. Display Results
        st.write("### AI Response:")
        st.success(response.content)
        
        # 4. Reference Attribution (GxP Transparency)
        sources = set([os.path.basename(doc.metadata['source']) for doc in results])
        st.info(f"📄 Sources consulted: {', '.join(sources)}")
