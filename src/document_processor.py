from __future__ import annotations
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)

from .utils import get_logger

logger = get_logger("document_processor")


#  Single-file loaders

def _load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.name))
    return docs


def _load_text(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.name))
    return docs


def _load_markdown(path: Path) -> List[Document]:
    loader = UnstructuredMarkdownLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.name))
    return docs


def _load_docx(path: Path) -> List[Document]:
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.name))
    return docs


def _load_csv(path: Path) -> List[Document]:
    loader = CSVLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.name))
    return docs


def _load_file_by_extension(path: Path) -> List[Document]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix == ".txt":
        return _load_text(path)
    elif suffix == ".md":
        return _load_markdown(path)
    elif suffix == ".docx":
        return _load_docx(path)
    elif suffix == ".csv":
        return _load_csv(path)
    else:
        logger.warning("Skipping unsupported file type: %s", path)
        return []


#  Public API

def load_documents_from_folder(folder_path: str) -> List[Document]:
    base_path = Path(folder_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Folder does not exist: {base_path}")

    all_docs: List[Document] = []

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        docs = _load_file_by_extension(file_path)
        if docs:
            logger.info("Loaded %d documents from %s", len(docs), file_path.name)
            all_docs.extend(docs)

    logger.info("Total loaded documents: %d", len(all_docs))
    return all_docs


def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    if not docs:
        logger.warning("split_documents called with empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks.", len(docs), len(chunks))
    return chunks
