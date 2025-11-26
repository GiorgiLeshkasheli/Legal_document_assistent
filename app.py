import os
import json
from pathlib import Path
from typing import List, Optional
import streamlit as st
from langchain.schema import Document
from src.utils import load_environment, get_logger
from src.document_processor import load_documents_from_folder, split_documents
from src.vector_store import build_faiss_index, get_retriever, load_faiss_index
from src.chatbot import create_legal_rag_chain, answer_question


#  CONFIG & HELPERS

PROJECT_ROOT = Path(__file__).parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_documents"
UPLOADED_DOCS_DIR = PROJECT_ROOT / "data" / "uploaded_documents"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"


def ensure_upload_folder() -> Path:
    UPLOADED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADED_DOCS_DIR


def save_uploaded_files(uploaded_files) -> Path:
    folder = ensure_upload_folder()
    for file in uploaded_files:
        file_path = folder / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    return folder


def load_pipeline_from_vectorstore(logger) -> Optional[dict]:
    if not VECTORSTORE_DIR.exists():
        return None

    try:
        vector_store = load_faiss_index(VECTORSTORE_DIR)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error("Failed to load FAISS index from %s: %s", VECTORSTORE_DIR, e)
        return None

    retriever = get_retriever(vector_store, k=5)
    chain = create_legal_rag_chain(retriever)

    return {
        "docs": [],
        "chunks": [],
        "retriever": retriever,
        "chain": chain,
    }


def build_pipeline_from_folder(folder: Path, logger) -> Optional[dict]:
    if not folder.exists():
        st.error(f"Folder not found: {folder}")
        logger.error("Folder does not exist: %s", folder)
        return None

    logger.info("Loading documents from folder: %s", folder)
    docs: List[Document] = load_documents_from_folder(str(folder))
    chunks: List[Document] = split_documents(docs)

    if not chunks:
        st.error("No chunks were created from the documents.")
        logger.warning("No chunks created from folder: %s", folder)
        return None

    # Build FAISS from chunks, persist index, then build retriever + chain
    vector_store = build_faiss_index(chunks, persist_dir=VECTORSTORE_DIR)
    retriever = get_retriever(vector_store, k=5)
    chain = create_legal_rag_chain(retriever)

    return {
        "docs": docs,
        "chunks": chunks,
        "retriever": retriever,
        "chain": chain,
    }


# ---------- STREAMLIT APP ----------

def main():
    st.set_page_config(
        page_title="Legal Document RAG Assistant",
        page_icon="📚",
        layout="wide",
    )

    # Load env + logger
    load_environment()
    logger = get_logger()

    st.title("📚 Legal Document RAG Assistant")
    st.write(
        "Ask questions about your **legal documents**.\n\n"
        "The assistant will answer *only* based on the uploaded files and will show sources."
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    mode = st.sidebar.radio(
        "Choose document source:",
        options=["Use sample documents", "Upload your own documents"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")

    #  Session state init
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "docs_count" not in st.session_state:
        st.session_state.docs_count = 0
    if "chunks_count" not in st.session_state:
        st.session_state.chunks_count = 0
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list[(role, message)]
    if "auto_built" not in st.session_state:
        st.session_state.auto_built = False

    #  Document selection / upload
    selected_folder: Optional[Path] = None
    uploaded_files = None

    if mode == "Use sample documents":
        st.info(
            "Using sample legal documents from "
            f"`{SAMPLE_DOCS_DIR.relative_to(PROJECT_ROOT)}`"
        )
        selected_folder = SAMPLE_DOCS_DIR
    else:
        uploaded_files = st.file_uploader(
            "Upload one or more legal documents (PDF, DOCX, MD, CSV, TXT)",
            type=["pdf", "txt", "docx", "md", "csv"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) selected.")
        selected_folder = None  # will be set after saving

    #  SMART AUTO-BUILD + PERSISTENT VECTORSTORE

    if not st.session_state.pipeline_ready and not st.session_state.auto_built:
        # 1) First try to load an existing FAISS index from disk
        existing = load_pipeline_from_vectorstore(logger)
        if existing is not None:
            st.session_state.pipeline_ready = True
            st.session_state.chain = existing["chain"]
            st.session_state.auto_built = True

            st.sidebar.success("Loaded existing vector index ✅")
        else:
            # 2) If no index yet, auto-build from current mode if possible
            folder_for_auto: Optional[Path] = None

            if mode == "Use sample documents":
                folder_for_auto = SAMPLE_DOCS_DIR
            elif mode == "Upload your own documents" and uploaded_files:
                folder_for_auto = save_uploaded_files(uploaded_files)

            if folder_for_auto is not None:
                with st.spinner("Auto-building RAG pipeline from documents..."):
                    result = build_pipeline_from_folder(folder_for_auto, logger)

                if result is not None:
                    st.session_state.pipeline_ready = True
                    st.session_state.docs_count = len(result["docs"])
                    st.session_state.chunks_count = len(result["chunks"])
                    st.session_state.chain = result["chain"]
                    st.session_state.auto_built = True

                    st.success("RAG pipeline built automatically ✅")
                    st.sidebar.success("Pipeline ready ✅")
                    st.sidebar.write(f"📄 Documents: **{st.session_state.docs_count}**")
                    st.sidebar.write(f"🧩 Chunks: **{st.session_state.chunks_count}**")
                else:
                    st.warning("Automatic build failed. You can try using the rebuild button.")

    #  Manual Build / Rebuild Button

    if st.button("🔄 Build / Rebuild RAG Pipeline"):
        # We are now explicitly rebuilding → mark as built
        st.session_state.auto_built = True

        if mode == "Upload your own documents":
            if not uploaded_files:
                st.warning("Please upload at least one document first.")
                st.stop()

            folder = save_uploaded_files(uploaded_files)
            selected_folder = folder
            st.sidebar.write(f"📁 Using uploaded docs: `{folder.name}`")

        if mode == "Use sample documents":
            selected_folder = SAMPLE_DOCS_DIR

        if not selected_folder:
            st.error("No document folder selected or created.")
            st.stop()

        with st.spinner("Processing documents and building FAISS index..."):
            result = build_pipeline_from_folder(selected_folder, logger)

        if result is None:
            st.session_state.pipeline_ready = False
            st.error("Failed to build pipeline. Check logs for details.")
        else:
            st.session_state.pipeline_ready = True
            st.session_state.docs_count = len(result["docs"])
            st.session_state.chunks_count = len(result["chunks"])
            st.session_state.chain = result["chain"]

            st.success("RAG pipeline rebuilt successfully! ✅")
            st.sidebar.success("Pipeline ready ✅")
            st.sidebar.write(f"📄 Documents: **{st.session_state.docs_count}**")
            st.sidebar.write(f"🧩 Chunks: **{st.session_state.chunks_count}**")

    # Show current status
    if st.session_state.pipeline_ready:
        st.sidebar.success("Ready for questions")
        if st.session_state.docs_count:
            st.sidebar.write(f"📄 Documents: **{st.session_state.docs_count}**")
        if st.session_state.chunks_count:
            st.sidebar.write(f"🧩 Chunks: **{st.session_state.chunks_count}**")
    else:
        st.sidebar.warning("Pipeline not built yet")

    st.markdown("---")

    # Chat interface

    st.subheader("💬 Ask a legal question")

    if not st.session_state.pipeline_ready or st.session_state.chain is None:
        st.info("The RAG pipeline is not ready yet. Upload/build first.")
        return

    chain = st.session_state.chain

    # Show previous chat messages
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    user_question = st.chat_input("Type your question about the documents...")

    if user_question:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(chain, user_question)
                    answer_text = result.get("answer", "").strip()
                    sources = result.get("sources", [])

                    # Display answer
                    st.markdown(answer_text or "_No answer returned._")

                    # Display sources if available
                    if sources:
                        st.markdown("**Sources:**")
                        for src in sources:
                            st.markdown(f"- {src}")
                except Exception as e:
                    logger.exception("Error while answering question: %s", e)
                    st.error(f"Error while generating answer: {e}")
                    answer_text = f"Error: {e}"

        # Save assistant message
        st.session_state.chat_history.append(("assistant", answer_text or ""))

    # Document Summarization

    st.markdown("---")
    st.subheader("📝 Document summarization")

    if st.button("Summarize current document set"):
        with st.spinner("Summarizing documents..."):
            try:
                summary_result = answer_question(
                    chain,
                    (
                        "Provide a clear, structured summary of the main legal topics, "
                        "responsibilities, user rights, limitations, and key clauses "
                        "across all provided documents. Focus on the most important points "
                        "and keep it under 10 bullet points."
                    ),
                )
                summary_text = summary_result.get("answer", "").strip()
            except Exception as e:
                logger.exception("Error while summarizing documents: %s", e)
                summary_text = f"Error while summarizing: {e}"

        st.markdown("#### Summary of current legal corpus")
        st.markdown(summary_text or "_No summary produced._")

    #Chat History Export (TXT / MD / JSON)

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📥 Export Chat History")

        # Prepare data
        lines = [
            f"{role.upper()}: {message}"
            for role, message in st.session_state.chat_history
        ]
        txt_data = "\n\n".join(lines)

        md_lines = [
            f"**{role.capitalize()}:** {message}"
            for role, message in st.session_state.chat_history
        ]
        md_data = "\n\n".join(md_lines)

        json_list = [
            {"role": role, "message": message}
            for role, message in st.session_state.chat_history
        ]
        json_data = json.dumps(json_list, ensure_ascii=False, indent=2)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="⬇️ Download as TXT",
                data=txt_data,
                file_name="chat_history.txt",
                mime="text/plain",
                key="download_txt",
            )

        with col2:
            st.download_button(
                label="⬇️ Download as Markdown",
                data=md_data,
                file_name="chat_history.md",
                mime="text/markdown",
                key="download_md",
            )

        with col3:
            st.download_button(
                label="⬇️ Download as JSON",
                data=json_data,
                file_name="chat_history.json",
                mime="application/json",
                key="download_json",
            )

if __name__ == "__main__":
    main()
