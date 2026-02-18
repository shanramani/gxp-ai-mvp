import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. GxP UI HEADER & CONFIG ---
st.set_page_config(page_title="GxP AI MVP", layout="wide", page_icon="🛡️")

# --- 2. INITIALIZE SESSION STATE (Must be at the top!) ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Custom CSS for Pharma-grade UI
st.markdown("""
    <style>
    .stChatMessage { background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stBadge { background-color: #e1e4e8; color: #31333F; padding: 5px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ GxP-Validated AI Knowledge Assistant")
st.caption("Grounded on Official SOP Library | v1.9 (Audit-Ready)")

# --- 3. LOGIC: ENGINE & LLM ---
@st.cache_resource
def setup_engine():
    path = "knowledge/"
    if not os.path.exists(path): 
        os.makedirs(path)
    all_files = os.listdir(path)
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
    return Chroma.from_documents(documents=all_pages, embedding=embeddings)

def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

# --- 4. SIDEBAR: KNOWLEDGE STATUS & AUDIT TRAIL ---
with st.sidebar:
    st.header("📚 Library Status")
    path = "knowledge/"
    all_files = os.listdir(path) if os.path.exists(path) else []
    current_pdfs = [f for f in all_files if f.lower().endswith('.pdf')]
    st.success(f"**{len(current_pdfs)}** SOPs Online")
    for f in current_pdfs: 
        st.caption(f"📄 {f}")
    
    st.markdown("---")
    st.header("📜 Audit Trail")
    if st.session_state.logs:
        df_logs = pd.DataFrame(st.session_state.logs)
        st.download_button("📥 Export Audit Log", df_logs.to_csv(index=False).encode('utf-8'), f"audit_{datetime.date.today()}.csv", "text/csv")
    
    for entry in reversed(st.session_state.logs):
        with st.expander(f"🕒 {entry['timestamp']}"):
            st.write(f"**Action:** {entry['query']}\n**Source:** {entry['source_type']}")

# --- 5. MAIN CHAT DISPLAY ---
engine = setup_engine()

# Display existing chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. USER INPUT & AI LOGIC ---
if prompt := st.chat_input("Ask about the SOP library or specific procedures..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if engine:
        with st.spinner("Analyzing..."):
            # A. Search Content (Source 2)
            results = engine.similarity_search(prompt, k=6)
            context_blocks = [f"SOURCE: {os.path.basename(d.metadata.get('source', 'Unknown'))} (Page {d.metadata.get('page', 0)+1})\nCONTENT: {d.page_content}" for d in results]
            context_text = "\n\n---\n\n".join(context_blocks)
            
            # B. System Metadata (Source 1)
            sop_list_str = ", ".join(current_pdfs)
            
            # C. Execute LLM with Routing
            llm = get_llm()
            system_prompt = f"""
            You are a GxP Compliance Assistant. Sources:
            1. SYSTEM METADATA (Filenames): {sop_list_str}
            2. DOCUMENT CONTENT (Text inside PDFs): {context_text}

            RULES:
            - If asking about the library/files/sidebar/inventory, start with 'SOURCE_TYPE: METADATA'.
            - If asking about SOP procedures/content/details, start with 'SOURCE_TYPE: CONTENT'. Cite SOP and Page.
            
            Question: {prompt}
            """
            
            response = llm.invoke(system_prompt)
            raw_content = response.content
            
            # D. Parse Source Type and Show Reference Pill
            is_content_query = "SOURCE_TYPE: CONTENT" in raw_content
            source_display = "📑 Document Content" if is_content_query else "📂 System Metadata"
            clean_response = raw_content.replace("SOURCE_TYPE: CONTENT", "").replace("SOURCE_TYPE: METADATA", "").strip()

            with st.chat_message("assistant"):
                st.markdown(f"**{source_display}**")
                st.markdown(clean_response)
                
                # Verified Grounding pill logic
                if is_content_query and results:
                    sources = set([f"{os.path.basename(d.metadata['source'])} (p.{d.metadata['page']+1})" for d in results])
                    st.info(f"**Verified Grounding:** {', '.join(sources)}")
            
            # E. Update State & Logs
            st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
            st.session_state.logs.append({
                "user": "Shan",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": prompt,
                "source_type": source_display,
                "status": "Success"
            })
    else:
        st.error("Engine not ready. Please ensure your /knowledge folder contains PDFs.")
