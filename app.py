import streamlit as st
from langchain_groq import ChatGroq
import os
import datetime
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- GxP UI HEADER ---
st.set_page_config(page_title="GxP AI MVP", layout="wide", page_icon="🛡️")

# Custom CSS for Pharma-grade UI
st.markdown("""
    <style>
    .stChatMessage { background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stBadge { background-color: #e1e4e8; color: #31333F; padding: 5px; border-radius: 5px; }
    .main { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ GxP-Validated AI Knowledge Assistant")
st.caption("Grounded on Official SOP Library | v1.7 (Audit-Ready)")

# --- INITIALIZE SESSION STATE ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

# --- LOGIC: ENGINE & LLM ---
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
    vectorstore = Chroma.from_documents(documents=all_pages, embedding=embeddings)
    return vectorstore

def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=st.secrets["GROQ_API_KEY"],
        temperature=0
    )

# --- SIDEBAR: KNOWLEDGE STATUS & AUDIT TRAIL ---
with st.sidebar:
    st.header("📚 Library Status")
    path = "knowledge/"
    all_files = os.listdir(path) if os.path.exists(path) else []
    current_pdfs = [f for f in all_files if f.lower().endswith('.pdf')]
    
    st.success(f"**{len(current_pdfs)}** SOPs Online")
    for f in current_pdfs:
        st.caption(f"📄 {f}")
    
    st.markdown("---")
    st.header("📜 Audit Trail (21 CFR)")
    
    if st.session_state.logs:
        df_logs = pd.DataFrame(st.session_state.logs)
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit Log (CSV)",
            data=csv,
            file_name=f"audit_trail_{datetime.date.today()}.csv",
            mime="text/csv",
        )
    
    for entry in reversed(st.session_state.logs):
        with st.expander(f"🕒 {entry['timestamp']}"):
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Source:** {entry['source_type']}")

# --- MAIN CHAT INTERFACE ---
engine = setup_engine()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about the SOP library or specific procedures..."):
    # 1. Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if engine:
        with st.spinner("Analyzing request and routing to correct data source..."):
            # A. Search content (Source 2)
            results = engine.similarity_search(prompt, k=6)
            context_blocks = []
            for doc in results:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 0) + 1
                context_blocks.append(f"SOURCE: {source_name} (Page {page_num})\nCONTENT: {doc.page_content}")
            
            context_text = "\n\n---\n\n".join(context_blocks)
            
            # B. Get Metadata list (Source 1)
            sop_list_str = ", ".join(current_pdfs)
            
            # C. Routing Prompt Logic
            llm = get_llm()
            system_prompt = f"""
            You are a GxP Compliance Assistant. You have access to two data sources:
            
            1. SYSTEM METADATA: A list of the current SOP files: {sop_list_str}
            2. DOCUMENT CONTENT: Specific text retrieved from inside those files: {context_text}

            INSTRUCTIONS:
            - If the user asks about the library, what documents you have, or the sidebar, use SYSTEM METADATA. Start your answer with "SOURCE_TYPE: METADATA".
            - If the user asks about procedures, instructions, or details inside an SOP, use DOCUMENT CONTENT. Start your answer with "SOURCE_TYPE: CONTENT". Mention SOP names and Page Numbers.
            - If the answer is not available, say you don't know based on the grounded library.

            User Question: {prompt}
            """
            
            response = llm.invoke(system_prompt)
            raw_content = response.content
            
            # D. Parse Source Type for UI
            if "SOURCE_TYPE: METADATA" in raw_content:
                source_display = "📂 System Metadata (Library List)"
                clean_response = raw_content.replace("SOURCE_TYPE: METADATA", "").strip()
            else:
                source_display = "📑 Document Content (Inside SOPs)"
                clean_response = raw_content.replace("SOURCE_TYPE: CONTENT", "").strip()

            # 2. Display Assistant response
            with st.chat_message("assistant"):
                st.markdown(f"**{source_display}**")
                st.markdown(clean_response)
                
                # Show Verified Grounding pills if it was a content-based query
                if "CONTENT" in raw_content:
                    sources = set([f"{os.path.basename(d.metadata['source'])} (p.{d.metadata['page']+1})" for d in results])
                    st.info(f"**Verified Grounding:** {', '.join(sources)}")
            
            # 3. Save to History and Audit Trail
            st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
            st.session_state.logs.append({
                "user": "Shan (Lead)",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": prompt,
                "source_type": source_display,
                "status": "Success"
            })
    else:
        st.error("No PDFs detected in the 'knowledge/' folder. Please upload SOPs to continue.")
