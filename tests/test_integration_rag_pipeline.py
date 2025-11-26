from langchain_core.documents import Document
from src.document_processor import load_documents_from_folder, split_documents
from src.vector_store import build_faiss_index, get_retriever


def test_full_retrieval_pipeline(sample_text_documents):
    # Load raw documents from the temp fixture folder
    docs = load_documents_from_folder(str(sample_text_documents))
    assert len(docs) >= 2
    assert all(isinstance(d, Document) for d in docs)

    # Split into chunks
    chunks = split_documents(docs)
    assert len(chunks) >= len(docs)
    assert all(isinstance(c, Document) for c in chunks)

    # Build FAISS index
    vector_store = build_faiss_index(chunks)

    # Create retriever
    retriever = get_retriever(vector_store, k=2)

    # Do a semantic retrieval
    query = "jurisdiction and governing law"
    results = retriever.get_relevant_documents(query)

    assert isinstance(results, list)
    assert len(results) >= 1
    assert all(isinstance(r, Document) for r in results)

    # Check that some result text is relevant
    combined_text = " ".join(r.page_content.lower() for r in results)
    assert "jurisdiction" in combined_text or "governing law" in combined_text
