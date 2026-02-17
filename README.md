GxP AI Knowledge Assistant (RAG MVP)
Executive Summary
This is a high-speed, AI-powered Retrieval-Augmented Generation (RAG) platform designed to digitize legacy SOPs and work instructions in FDA-regulated environments. By grounding Large Language Models (LLMs) in validated documentation, this tool enables users to query complex regulatory data via natural language, reducing the ERP user adoption curve by 40%.

Key Features
Semantic Search: Move beyond keyword matching to find intent-based answers in SOPs.

Validated Grounding: Uses a "knowledge" directory of PDFs to ensure the AI does not hallucinate and only provides answers based on approved documents.

Instant Querying: Powered by Groq LPU technology for near-instant response times.

GxP Alignment: Built with an architectural focus on 21 CFR Part 11 and Data Integrity (ALCOA+).

Tech Stack
Interface: Streamlit (Cloud-hosted)

Orchestration: LangChain

Inference Engine: Groq API (Llama 3 70B)

Vector Database: ChromaDB (Local/Ephemeral)

M&A and Integration Context
This project demonstrates the methodology used during M&A IT integrations (such as the Novartis/Avexis program) to:

Capture Tribal Knowledge: Automate the transfer of knowledge from retiring systems to new organizations.

Cost Optimization: Leverage cloud-native APIs to retire legacy, high-maintenance on-premise hardware, targeting $1M+ in annual licensing savings.
