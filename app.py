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

# Custom CSS for a professional "Pharma" look
st.markdown("""
    <style>
    .stChatMessage { background-color: #f0f2f6; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    .stBadge { background-color: #e1e4e8; }
    .main { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ GxP-Validated AI Knowledge Assistant")
st.caption("Standard Operating Procedure (SOP) Grounded Intelligence | v1.5")

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
    # Case-insensitive PDF detection
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
    all_files = os.listdir(path)
    current_pdfs = [f for f in all_files if f.lower().endswith('.pdf')]
    
    st.success(f"**{len(current_pdfs)}** SOPs Online")
    for f in current_pdfs:
        st.caption(f"📄 {f}")
    
    st.markdown("---")
    st.header("📜 Audit Trail (21 CFR)")
    
    # Export Audit Log Button
    if st.session_state.logs:
        df_logs = pd.DataFrame(st.session_state.logs)
        csv = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Audit Log (CSV)",
            data=csv,
            file_name=f"audit_trail_{datetime.date.today()}.csv",
            mime="text/csv",
        )
    
    # Display individual log entries
    for entry in reversed(st.session_state.logs):
        with st.expander(f"🕒 {entry['timestamp']}"):
            st.write(f"**User:** {entry['user']}")
            st.write(f"**Query:** {entry['query']}")
            st.write(f"**Status:** {entry['status']}")

# --- MAIN CHAT INTERFACE ---
engine = setup_engine()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the SOP library..."):
    # 1. Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. RAG Logic
    if engine:
        with st.spinner("Analyzing validated sources..."):
            # Search for top 6 relevant snippets
            results = engine.similarity_search(prompt, k=6)
            
            # Build context with Page Numbers for citations
            context_blocks = []
            for doc in results:
                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 0) + 1
                context_blocks.append(f"SOURCE: {source_name} (Page {page_num})\nCONTENT: {doc.page_content}")
            
            context_text = "\n\n---\n\n".join(context_blocks)
            
            # Get the list of all indexed files for system grounding
            indexed_docs = ", ".join(current_pdfs)
            
            # 3. Get AI Response with strict GxP instructions
            llm = get_llm()
            system_prompt = f"""
            You are a precise GxP Compliance Assistant. 
            The total grounded library available to you consists of: {indexed_docs}

            Use the following retrieved context to answer. 
            ALWAYS mention the SOP name and Page Number in your answer.
            If the answer is not in the context, state that you cannot find the info in the current validated library.

            Context:
            {context_text}

            Question: {prompt}
            """
            
            response = llm.invoke(system_prompt)
            
            # 4. Display Assistant response
            with st.chat_message("assistant"):
                st.markdown(response.content)
                # Show verified Source Pills for auditability
                sources = set([f"{os.path.basename(d.metadata['source'])} (p.{d.metadata['page']+1})" for d in results])
                st.info(f"**Verified Grounding:** {', '.join(sources)}")
            
            st.session_state.chat_history.append({"role": "assistant", "content": response.content})

            # 5. Log the successful event in Audit Trail
            st.session_state.logs.append({
                "user": "Shan (Lead)",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": prompt,
                "status": "Success"
            })
    else:
        st.error("Engine not initialized. Please ensure PDFs are in the /knowledge folder.")
