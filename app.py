import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide", page_icon="🛡️")

# Custom CSS for a "cleaner" Pharma look
st.markdown("""
    <style>
    .stChatMessage { background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stBadge { background-color: #e1e4e8; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ GxP-Validated AI Knowledge Assistant")
st.caption("Version 1.2 | Grounded on Official SOP Library")

# --- INITIALIZE SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- SIDEBAR: AUDIT & STATUS ---
with st.sidebar:
    st.header("📚 Library Status")
    path = "knowledge/"
    if not os.path.exists(path): os.makedirs(path)
    all_files = os.listdir(path)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    st.success(f"**{len(pdf_files)}** SOPs Online")
    for f in pdf_files:
        st.caption(f"📄 {f}")
    
    st.markdown("---")
    st.header("📜 Audit Trail (21 CFR)")
    for entry in st.session_state.logs:
        st.caption(f"**{entry['time']}**: {entry['text']}")

# --- LOGIC: ENGINE & LLM ---
@st.cache_resource
def setup_engine():
    path = "knowledge/"
    all_files = os.listdir(path)
    pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
    if not pdf_files: return None
    
    all_pages = []
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(path, pdf))
        all_pages.extend(loader.load())

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=all_pages, embedding=embeddings)

def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

engine = setup_engine()

# --- CHAT INTERFACE (The "ChatGPT" Look) ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about SOP details (e.g., 'What is the deviation process?')"):
    # 1. Add user message to chat and logs
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({"time": now, "text": prompt})

    # 2. RAG Logic with Citations
    if engine:
        with st.spinner("Searching validated sources..."):
            results = engine.similarity_search(prompt, k=5)
            
            # Build context with Page Numbers
            context_blocks = []
            for doc in results:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 0) + 1  # Index starts at 0
                context_blocks.append(f"SOURCE: {source_name} (Page {page_num})\nCONTENT: {doc.page_content}")
            
            context = "\n\n---\n\n".join(context_blocks)
            
            # 3. Get AI Response
            llm = get_llm()
            full_prompt = f"""
            You are a precise GxP Compliance Officer. Use the context below to answer.
            ALWAYS mention the SOP name and Page Number in your answer when possible.
            If the answer is not in the context, say you don't know based on current SOPs.

            Context:
            {context}

            Question: {prompt}
            """
            
            response = llm.invoke(full_prompt)
            
            # 4. Display Assistant response
            with st.chat_message("assistant"):
                st.markdown(response.content)
                # Show explicit Source Pills
                sources = set([f"{os.path.basename(d.metadata['source'])} (p.{d.metadata['page']+1})" for d in results])
                st.info(f"**Verified Sources:** {', '.join(sources)}")
            
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})
