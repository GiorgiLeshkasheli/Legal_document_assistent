import pytest
from pathlib import Path
from langchain_core.documents import Document


# sample folder with multiple small .txt legal documents

@pytest.fixture
def sample_text_documents(tmp_path: Path) -> Path:
    folder = tmp_path / "sample_docs"
    folder.mkdir(parents=True, exist_ok=True)

    # doc1
    (folder / "doc1.txt").write_text(
        "This is a sample legal document.\n"
        "It discusses jurisdiction and governing law between the parties."
    )

    # doc2
    (folder / "doc2.txt").write_text(
        "This document includes termination, suspension, and liability clauses.\n"
        "Services may be suspended under certain contractual conditions."
    )

    # doc3
    (folder / "doc3.txt").write_text(
        "Confidentiality obligations apply to both parties.\n"
        "This section covers confidentiality and data usage."
    )

    return folder


# simple list of Document objects

@pytest.fixture
def small_document_list() -> list[Document]:

    return [
        Document(
            page_content="Jurisdiction clause. Governing law applies.",
            metadata={"source": "test_doc1.txt", "page": 0},
        ),
        Document(
            page_content="Termination clause. Rights and responsibilities.",
            metadata={"source": "test_doc2.txt", "page": 1},
        ),
    ]
