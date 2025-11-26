from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from .utils import get_logger, load_environment

logger = get_logger("chatbot")


# LLM + Memory setup

def get_llm() -> ChatGoogleGenerativeAI:
    load_environment()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")

    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

    logger.info(f"Using Gemini model: {model_name}")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True,
    )
    return llm


def get_memory() -> ConversationBufferMemory:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return memory


# Prompt & QA Chain

def build_legal_prompt() -> PromptTemplate:
    template = """
You are a helpful Legal Document Assistant.

You answer questions ONLY based on the provided legal documents.
If the answer is not clearly supported by the documents, you MUST say:
"I cannot find a clear answer to this question in the provided documents."

You must:
- Use precise legal language when possible.
- Summarize relevant clauses instead of copy-pasting long text.
- At the end of the answer, add a short "Sources:" section mentioning document names and (if available) page numbers.

--------------------
[CONTEXT - EXCERPTS FROM DOCUMENTS]
{context}
--------------------
[CHAT HISTORY]
{chat_history}
--------------------
[QUESTION]
{question}
--------------------

Think step by step, and then give a clear, structured answer.
If multiple interpretations exist, mention them.
If no relevant information exists, clearly say so.
    """

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"],
    )


def create_legal_rag_chain(retriever) -> ConversationalRetrievalChain:
    llm = get_llm()
    memory = get_memory()
    prompt = build_legal_prompt()

    logger.info("Building ConversationalRetrievalChain for Legal Assistant...")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    return chain


# Summarization

def summarize_documents(
    chunks: List[Document],
    focus: Optional[str] = None,
    max_chunks: int = 25,
) -> str:
    if not chunks:
        raise ValueError("No document chunks available to summarize.")

    llm = get_llm()

    # Truncate to first N chunks
    selected = chunks[:max_chunks]
    context_text = "\n\n".join(doc.page_content for doc in selected)

    if focus:
        task = (
            "Write a clear, concise summary of the legal content, "
            f"focusing especially on: {focus}."
        )
    else:
        task = (
            "Write a clear, concise summary of the main legal topics, parties, "
            "rights, obligations, limitations, and key risks in these documents."
        )

    prompt = f"""
You are a legal assistant.

Based ONLY on the text below, produce a structured summary that a lawyer or
policy analyst could quickly read to understand the key points.

Use short sections and bullet points where helpful.

-------------------- TEXT START --------------------
{context_text}
-------------------- TEXT END --------------------

Task: {task}
"""

    logger.info(
        "Running summarization on %d chunks (max_chunks=%d, focus=%s)",
        len(selected),
        max_chunks,
        focus or "None",
    )

    response = llm.invoke(prompt)
    try:
        return response.content
    except AttributeError:
        return str(response)


# High-level QA helper

def _extract_sources_from_docs(docs: List[Document]) -> List[str]:
    sources = []

    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None) or d.metadata.get("page_number", None)

        if isinstance(page, int):
            label = f"{src} (page {page + 1})"
        elif page is not None:
            label = f"{src} (page {page})"
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


def answer_question(
    chain: ConversationalRetrievalChain,
    question: str,
) -> Dict[str, Any]:
    if not question.strip():
        raise ValueError("Question must not be empty.")

    logger.info(f"User question: {question}")

    result = chain({"question": question})

    answer = result.get("answer", "")
    source_docs: List[Document] = result.get("source_documents", [])

    sources = _extract_sources_from_docs(source_docs)

    return {
        "answer": answer,
        "sources": sources,
        "raw_source_documents": source_docs,
    }
