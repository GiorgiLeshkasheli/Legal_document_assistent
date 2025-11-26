from src.utils import load_environment, get_logger
from src.document_processor import load_documents_from_folder, split_documents
from src.vector_store import build_faiss_index, get_retriever
from src.chatbot import create_legal_rag_chain, answer_question


def main():
    load_environment()
    logger = get_logger()

    logger.info("Starting Legal Document RAG Assistant...")

    # Load & split docs
    docs = load_documents_from_folder("data/sample_documents")
    chunks = split_documents(docs)

    print("\n----- SUMMARY -----")
    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        logger.warning("No chunks found. Exiting.")
        return

    # Vector store + retriever
    vector_store = build_faiss_index(chunks)
    retriever = get_retriever(vector_store, k=4)

    # Build RAG chatbot chain
    chain = create_legal_rag_chain(retriever)

    # Ask a sample legal question
    question = "Under which conditions can Google change or stop providing the services according to these terms?"
    result = answer_question(chain, question)

    print("\n----- CHATBOT ANSWER -----")
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f"- {src}")


if __name__ == "__main__":
    main()
