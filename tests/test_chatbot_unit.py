from typing import List
from langchain_core.documents import Document
from src.chatbot import (
    build_legal_prompt,
    _extract_sources_from_docs,
    answer_question,
    create_legal_rag_chain,
)
from langchain_core.retrievers import BaseRetriever


class DummyRetriever(BaseRetriever):

    def __init__(self, docs: List[Document] | None = None):
        super().__init__()
        self._docs = docs or []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # In tests we just return the fixed docs, regardless of query
        return self._docs


class FakeChain:

    def __init__(self, answer: str, docs: List[Document]):
        self._answer = answer
        self._docs = docs

    def __call__(self, inputs):
        return {
            "answer": self._answer,
            "source_documents": self._docs,
        }


def test_build_legal_prompt_structure():
    prompt = build_legal_prompt()
    assert {"context", "chat_history", "question"} == set(prompt.input_variables)
    assert "ONLY based on the provided legal documents" in prompt.template


def _extract_sources_from_docs(docs: List[Document]) -> List[str]:
    sources: List[str] = []

    for d in docs:
        src = d.metadata.get("source", "unknown")

        label: str
        if "page" in d.metadata and isinstance(d.metadata["page"], int):
            page_idx = d.metadata["page"]
            label = f"{src} (page {page_idx + 1})"
        elif "page_number" in d.metadata:
            page_num = d.metadata["page_number"]
            label = f"{src} (page {page_num})"
        else:
            label = src

        sources.append(label)

    # Remove duplicates while preserving order
    seen = set()
    unique_sources: List[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    return unique_sources



def test_answer_question_happy_path():
    docs = [
        Document(page_content="Some legal content", metadata={"source": "file1.txt"}),
        Document(page_content="More content", metadata={"source": "file2.txt"}),
    ]
    chain = FakeChain(answer="This is a test answer", docs=docs)

    result = answer_question(chain, "What is in the documents?")
    assert "answer" in result
    assert "sources" in result
    assert "raw_source_documents" in result

    assert result["answer"] == "This is a test answer"
    assert len(result["sources"]) == 2
    assert all(isinstance(d, Document) for d in result["raw_source_documents"])


def test_answer_question_raises_on_empty_query():
    docs = []
    chain = FakeChain(answer="irrelevant", docs=docs)
    try:
        answer_question(chain, "   ")
        raise AssertionError("Expected ValueError for empty question")
    except ValueError:
        pass


def test_create_legal_rag_chain_with_dummy_retriever():
    retriever = DummyRetriever(docs=[])
    chain = create_legal_rag_chain(retriever)

    assert hasattr(chain, "memory")
    assert hasattr(chain, "combine_docs_chain")
