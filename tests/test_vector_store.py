from langchain_core.documents import Document
from src.vector_store import build_faiss_index, get_retriever


def test_build_faiss_index_creates_store(small_document_list):
    vector_store = build_faiss_index(small_document_list)

    # Basic sanity checks
    assert vector_store is not None

    # A simple similarity search should return at least one doc
    results = vector_store.similarity_search("jurisdiction", k=1)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert isinstance(results[0], Document)


def test_get_retriever_works(small_document_list):
    vector_store = build_faiss_index(small_document_list)
    retriever = get_retriever(vector_store, k=2)

    # Retriever should give relevant documents
    retrieved_docs = retriever.get_relevant_documents("termination rights")
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) >= 1
    assert isinstance(retrieved_docs[0], Document)
