📚 Legal Document RAG Assistant
Overview

The Legal Document RAG Assistant is a powerful multi-document Retrieval-Augmented Generation (RAG) chatbot that can analyze legal documents—such as contracts, terms of service, policies, and regulations—and answer questions strictly based on their content.

This system uses semantic search (FAISS) + large language models (Google Gemini) to provide grounded, source-cited, legally-precise answers.

Use case examples:

Lawyers reviewing large contracts

Companies analyzing vendor agreements

Government teams reviewing legal regulations

Students learning legal interpretation

✨ Features
📁 Multi-Document Support

Users can load multiple files simultaneously.
Supported formats:

PDF

TXT

DOCX

Markdown (.md)

CSV

✂️ Smart Chunking

Documents are automatically split into optimally sized chunks for retrieval.

🔍 Semantic Retrieval (FAISS)

Uses Sentence-Transformers (all-MiniLM-L6-v2)

Fast vector similarity search

FAISS index is persisted to disk and auto-loaded on next run

🤖 Legal Question-Answering

LLM: Google Gemini (gemini-1.5-flash by default)

Based on ConversationalRetrievalChain

Strict grounding: If the answer is not in the documents, the assistant must say so

💬 Conversation Memory

Powered by ConversationBufferMemory for natural multi-turn dialogue.

📚 Source Citations

Every answer includes:

Document name

Page number (if available)

📝 Document Summarization

Button: "Summarize current document set"
Produces a structured summary of all uploaded documents.

💾 Chat History Export

Users can save their conversation in three formats:

TXT

Markdown

JSON

🌐 Streamlit UI

Modern, clean interface with:

Auto-building pipeline

Upload panel

Chat interface

Status indicators

Summarization + export buttons

🧪 Full Test Suite

Includes:

Unit tests for loading & splitting documents

Vector store building

Retrieval pipeline

Chatbot source extraction

Integration test (FAISS + retriever + RAG)
Run with pytest.

🧠 Tech Stack

Python 3.12

LangChain / LCEL

Google Gemini API (langchain-google-genai)

Sentence-Transformers embeddings

FAISS vector database

Streamlit

PyPDF / python-docx / Markdown / CSV loaders

pytest test suite

📂 Project Structure
rag-legal-assistant/
├── app.py                     # Streamlit UI
├── main.py                    # CLI interface (optional)
├── src/
│   ├── __init__.py
│   ├── chatbot.py             # LLM, prompts, RAG chain
│   ├── document_processor.py  # Loaders + chunking
│   ├── vector_store.py        # FAISS build/load + retriever
│   └── utils.py               # Logging, env helper
│
├── data/
│   ├── sample_documents/      # Preloaded legal documents
│   ├── uploaded_documents/    # User uploads
│   └── vectorstore/           # Persisted FAISS index
│
├── tests/                     # Full pytest suite
│   ├── conftest.py
│   ├── test_chatbot_unit.py
│   ├── test_document_processor.py
│   ├── test_vector_store.py
│   └── test_integration_rag_pipeline.py
│
├── .env                       # User's API keys (ignored)
├── .env.example               # Template for environment variables
├── requirements.txt
├── README.md
└── EXPLANATION.md

🚀 Installation
1. Install dependencies
pip install -r requirements.txt

2. Create a .env file
GEMINI_API_KEY=your_key_here
GEMINI_MODEL_NAME=gemini-1.5-flash
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

3. Run the Streamlit app
streamlit run app.py

🧪 Running Tests
pytest -q

💡 Summary

The Legal Document RAG Assistant is a complete solution for searching and understanding legal documents using modern AI. It includes:

✔ Multi-format document ingestion
✔ FAISS vector search
✔ Google Gemini-powered legal Q&A
✔ Automatic document summarization
✔ Chat export
✔ Persistent indexing
✔ Strong modular architecture
✔ Full test suite