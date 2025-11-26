from langchain_core.documents import Document
from src.document_processor import load_documents_from_folder, split_documents


def test_load_documents_from_folder_returns_documents(sample_text_documents):
    docs = load_documents_from_folder(str(sample_text_documents))

    assert isinstance(docs, list)
    assert len(docs) >= 2  # we wrote 2 files
    assert all(isinstance(d, Document) for d in docs)

    # At least one document should contain a known string
    contents = " ".join(d.page_content for d in docs).lower()
    assert "jurisdiction" in contents
    assert "suspended" in contents


def test_split_documents_returns_chunks(sample_text_documents):
    docs = load_documents_from_folder(str(sample_text_documents))
    chunks = split_documents(docs)

    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    # For short docs, chunks will often be >= docs count (or equal)
    assert len(chunks) >= len(docs)

    # Ensure content is non-empty
    assert all(c.page_content.strip() for c in chunks)
