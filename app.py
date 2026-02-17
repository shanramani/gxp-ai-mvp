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

# --- INITIALIZE SESSION STATE ---
if 'logs' not in st.session_state:
    st.session_state.logs = []

st.markdown("---")

# --- SIDEBAR: KNOWLEDGE STATUS & AUDIT TRAIL ---
with st.sidebar:
    st.header("📚 Knowledge Base Status")
    path = "knowledge/"
    if not os.path.exists(path):
        os.makedirs(path)
    
    # FIX: Look for both .pdf and .PDF
    all_files = os.listdir(path)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    
    st.write(f"Total SOPs Indexed: **{len(pdf_files)}**")
    for f in pdf_files:
        st.write(f"- {f}")
    
    st.markdown("---")
    st.header("📜 Audit Trail")
    for entry in st.session_state.logs:
        st.caption(f"{entry['time']}: {entry['text']}")

# --- LOGIC: INITIALIZE THE BRAIN ---
@st.cache_resource
def setup_engine():
    path = "knowledge/"
    all_files = os.listdir(path)
    # FIX: Ensure the engine actually loads the uppercase .PDF files too
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        return None
    
    all_pages = []
    for pdf in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(path, pdf))
            all_pages.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {pdf}: {e}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=all_pages, embedding=embeddings)
    return vectorstore

# --- LOGIC: CONNECT TO GROQ BRAIN ---
def get_llm():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        temperature=0
    )

# --- RUNNING THE APP ---
engine = setup_engine()
user_input = st.text_input("Ask a question about your SOP library:")

if user_input and engine:
    # 1. Properly Log activity
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": now, "text": user_input})
    
    with st.spinner("Analyzing all SOPs..."):
        # 2. Search & Answer (k=6 for better cross-document coverage)
        results = engine.similarity_search(user_input, k=6)
        context = "\n\n---\n\n".join([doc.page_content for doc in results])
        
        llm = get_llm()
        prompt = f"""
        You are a GxP Compliance Assistant. Use the following context from multiple SOPs to answer the question. 
        If the documents contain conflicting info, highlight both.
        
        Context:
        {context}
        
        Question: {user_input}
        
        Answer:"""
        
        response = llm.invoke(prompt)
        
        # 3. Display Results
        st.write("### AI Response:")
        st.success(response.content)
        
        # 4. Reference Attribution
        sources = set([os.path.basename(doc.metadata['source']) for doc in results])
        st.info(f"📄 Sources consulted: {', '.join(sources)}")
