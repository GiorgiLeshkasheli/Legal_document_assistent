from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .utils import get_logger

logger = get_logger("vector_store")


def get_embedding_model() -> HuggingFaceEmbeddings:
    model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    logger.info(f"Using embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings


def build_faiss_index(
    docs: List[Document],
    persist_dir: Optional[str | Path] = None,
) -> FAISS:
    if not docs:
        raise ValueError("No documents provided to build FAISS index.")

    embeddings = get_embedding_model()

    logger.info(f"Building FAISS index from {len(docs)} chunks...")
    vector_store = FAISS.from_documents(docs, embeddings)
    logger.info("FAISS index built successfully.")

    if persist_dir is not None:
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving FAISS index to: {persist_path}")
        vector_store.save_local(str(persist_path))

    return vector_store


def load_faiss_index(
    persist_dir: str | Path,
) -> FAISS:
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(f"FAISS index folder does not exist: {persist_path}")

    embeddings = get_embedding_model()
    logger.info(f"Loading FAISS index from: {persist_path}")
    vector_store = FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,  # local dev only
    )
    return vector_store


def get_retriever(vector_store: FAISS, k: int = 4):
    return vector_store.as_retriever(search_kwargs={"k": k})
