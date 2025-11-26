1. Use Case Choice and Problem Statement

For this project, I developed a Legal Document RAG Assistant — an AI system that helps users understand, analyze, and query legal documents such as Terms of Service, contracts, privacy policies, and regulations.

Why this use case?

Legal texts are long, dense, and difficult to understand.

Important clauses are hidden inside dozens of pages.

Manual search is slow and error-prone.

Keyword search (Ctrl+F) is not enough.

Lawyers, businesses, and students frequently need fast clause-level answers.

What problem does the system solve?

The assistant provides a multi-document question-answering system that:

accepts multiple legal documents (PDF, DOCX, MD, CSV, TXT)

processes and splits them into semantically meaningful chunks

stores all chunks inside a vector database

retrieves the most relevant sections for any question

answers using a Google Gemini LLM

cites exact document names and page numbers

offers document summarization

allows export of the chat history

This eliminates the need to manually browse long legal documents.

2. System Architecture Overview

The system follows a classical Retrieval-Augmented Generation (RAG) architecture using LangChain, Sentence-Transformers, FAISS, and Google Gemini.

2.1 Document Loading

The system supports multiple file formats:

PDF

TXT

DOCX

Markdown (.md)

CSV

The loader extracts text and assigns metadata:

document name

page number (if applicable)

Metadata is preserved throughout the pipeline for source citation.

2.2 Text Splitting (Chunking)

Documents are split into overlapping semantic chunks using LangChain's RecursiveCharacterTextSplitter.

Why chunking?

improves retrieval precision

reduces irrelevant context

enables clause-level analysis

prevents the LLM from receiving overly large contexts

Each chunk retains metadata (source + page), essential for legal transparency.

2.3 Embedding Generation & Vector Store (FAISS)

Each chunk is encoded with:

sentence-transformers/all-MiniLM-L6-v2

Reasons for choosing this embedding model:

free and works locally

lightweight and fast

high semantic quality

no API cost

Embeddings are stored inside a FAISS index, which is:

fast

scalable

local and offline

ideal for RAG systems

FAISS is persisted to disk, so the app:

loads the index instantly on next startup

doesn’t reprocess documents unless requested

2.4 Retrieval + LLM Integration

When a user asks a question:

The question is embedded.

FAISS retrieves the most relevant document chunks.

These chunks are passed as context to the LLM.

The LLM produces a grounded answer.

The system displays:

answer

extracted reasoning

exact sources (document + page)

The LLM used:

Google Gemini (gemini-1.5-flash)
via langchain-google-genai.

The RAG pipeline uses:

ConversationalRetrievalChain

ConversationBufferMemory

This allows natural multi-turn dialogue referencing previous messages.

3. Design Decisions
3.1 LLM Choice: Google Gemini

Reasons:

extremely fast

free-tier available

strong performance on reasoning tasks

easy integration with LangChain 0.2+

reliable response formatting

3.2 Embedding Model Choice

all-MiniLM-L6-v2 was selected because:

excellent performance for semantic search

low resource usage

no need for external API

ideal for legal RAG tasks

3.3 Vector Database: FAISS

Chosen because:

open-source

optimized for similarity search

works offline

highly reliable for local RAG systems

supports persistence between sessions

3.4 Modular Architecture

The project uses a clean, maintainable structure:

src/
│ chatbot.py           – RAG chain + prompts + Gemini integration
│ document_processor.py – multi-format loaders + chunking
│ vector_store.py       – FAISS build/load + retriever construction
│ utils.py              – logging + environment loader
app.py                  – Streamlit UI


This makes the system easy to extend with new features.

4. Challenges and How I Solved Them
4.1 Gemini Model Version Errors

Initially, Gemini returned 404 errors due to outdated model names.

Solution: Switched to the stable gemini-1.5-flash model and updated dependencies.

4.2 LangChain Version Conflicts

LangChain evolves rapidly, causing breaking changes.

Solution:

Upgraded to LangChain 0.2+

Replaced deprecated imports

Adjusted retriever + chain constructors

4.3 Complex Document Handling

Ensuring multi-format compatibility (PDF/DOCX/MD/CSV) required testing several loaders.

Used:

PyPDF

python-docx

markdown-to-text conversion

CSV column aggregation

4.4 Ensuring Persistent Vectorstore

To avoid rebuilding embeddings every time, FAISS persistence was implemented.

This required reorganizing vector_store.py to support:

loading index

rebuilding when needed

Now the system autoloads the index if it already exists.

5. Additional Features Implemented

The following features were added beyond the original project concept:

✔ Streamlit UI

A clean, interactive user interface with:

file upload

pipeline building

status indicators

chat window

source citations

summarization

export buttons

✔ Document Summarization

Users can summarize all processed documents in one click.

✔ Chat History Export

Export in:

TXT

Markdown

JSON

✔ Persistent Vectorstore

FAISS index is saved and auto-loaded.

✔ Multi-Format Document Support

PDF, TXT, DOCX, Markdown, CSV.

These features significantly enhance real-world usability.

6. Current Limitations

Despite being highly functional, a few limitations remain:

OCR is not implemented (scanned PDFs may fail)

Retrieval confidence scores are not displayed yet

Multilingual documents not tested extensively

No advanced citation highlighting inside the UI

These limitations are acceptable for the current scope.

7. Future Improvements

If further development continues, the following upgrades are planned:

🔹 Chunk-level similarity score display

Show how relevant each chunk is.

🔹 OCR support for scanned PDFs

Using Tesseract or PaddleOCR.

🔹 Clause extraction and classification

E.g. "Find liability clauses" or "Detect GDPR violations".

🔹 Multi-query retrieval

To improve answer robustness.

🔹 Chat session saving & loading

Full session persistence.

8. Conclusion

The system successfully delivers a robust multi-document legal RAG assistant with:

multi-format document ingestion

intelligent chunking

persistent FAISS vectorstore

HuggingFace embeddings

Gemini-powered RAG

conversational memory

document summarization

chat export

modular software design

full pytest test suite

The result is a complete, scalable, and realistic AI-powered tool for legal document analysis—far beyond a simple demo and close to a production-ready system.