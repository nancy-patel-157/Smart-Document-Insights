"""RAG engine: PDF loading, indexing, retrieval and all LLM-powered features.

This module has no Streamlit code, so it can be tested with fake LLMs and
fake embeddings (see tests/).
"""

import io
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import prompts
from helpers import (
    contains_date,
    format_history,
    sample_evenly,
    source_label,
    split_into_sections,
    take_until,
)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Up to this many characters the summary is written in one call;
# longer documents are summarised section by section (map-reduce).
DIRECT_SUMMARY_LIMIT = 15000


# ------------------------------------------------------------------ models
def get_llm(api_key, temperature=0.2, model=None):
    """Gemini chat model. Imported lazily so tests don't need the package."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODEL, google_api_key=api_key, temperature=temperature
    )


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def to_text(message):
    """Plain text from an LLM response (newer Gemini SDKs may return content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------- indexing
def load_pdfs(files):
    """Read PDFs into one Document per page, keeping file name and page number.

    ``files`` is a list of (file_name, file_bytes).
    Returns (documents, total_page_count, warnings).
    """
    documents, page_count, warnings = [], 0, []
    for name, data in files:
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(""):
                warnings.append(f"{name}: password-protected, skipped.")
                continue
            pages = reader.pages
        except Exception as error:  # corrupt or unsupported file
            warnings.append(f"{name}: could not be read ({error}).")
            continue

        page_count += len(pages)
        empty_pages = 0
        for number, page in enumerate(pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                documents.append(
                    Document(page_content=text, metadata={"source": name, "page": number})
                )
            else:
                empty_pages += 1
        if empty_pages == len(pages) and pages:
            warnings.append(f"{name}: no selectable text (scanned PDF?). Run OCR first.")
        elif empty_pages:
            warnings.append(f"{name}: {empty_pages} page(s) had no text and were skipped.")
    return documents, page_count, warnings


def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split page documents into chunks. Page metadata is copied to every chunk."""
    chunk_overlap = min(chunk_overlap, chunk_size // 2)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_index(chunks, embeddings):
    return FAISS.from_documents(chunks, embedding=embeddings)


def full_text(documents):
    return "\n".join(doc.page_content for doc in documents)


# --------------------------------------------------------------- retrieval
def format_context(documents):
    return "\n\n".join(
        f"[{source_label(doc.metadata)}]\n{doc.page_content.strip()}" for doc in documents
    )


def dedupe(documents):
    seen, unique = set(), []
    for doc in documents:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:200])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def multi_query_retrieve(store, queries, k_each=4, limit=14):
    """Run several searches and merge the results (used by the extractors)."""
    results = []
    for query in queries:
        results.extend(store.similarity_search(query, k=k_each))
    return dedupe(results)[:limit]


# ---------------------------------------------------------------- features
def condense_question(llm, question, history):
    """Rewrite a follow-up ("explain it more") into a standalone question."""
    if not history:
        return question
    prompt = prompts.CONDENSE_PROMPT.format(history=format_history(history), question=question)
    rewritten = to_text(llm.invoke(prompt)).strip()
    return rewritten or question


def answer_question(llm, store, question, history=None, language="English", k=4):
    """Chat answer grounded in retrieved chunks, with page citations and memory."""
    standalone = condense_question(llm, question, history or [])
    sources = store.similarity_search(standalone, k=k)
    prompt = prompts.QA_PROMPT.format(
        context=format_context(sources),
        question=standalone,
        not_found=prompts.NOT_FOUND.get(language, prompts.NOT_FOUND["English"]),
        math_rule=prompts.MATH_RULE,
        language_rule=prompts.language_rule(language),
    )
    answer = to_text(llm.invoke(prompt)).strip()
    return {"answer": answer, "sources": sources, "standalone_question": standalone}


def summarize(llm, text, language="English"):
    """Executive summary + 10 key points. Long documents use map-reduce so the
    whole document is covered, not just the first few pages."""
    if len(text) > DIRECT_SUMMARY_LIMIT:
        sections = split_into_sections(text, section_chars=12000, max_sections=10)
        map_prompts = [
            prompts.MAP_PROMPT.format(index=i, total=len(sections), text=section)
            for i, section in enumerate(sections, start=1)
        ]
        partials = [to_text(r) for r in llm.batch(map_prompts)]
        text = "\n\n".join(
            f"Notes from part {i}:\n{p}" for i, p in enumerate(partials, start=1)
        )
    prompt = prompts.SUMMARY_PROMPT.format(
        text=text, math_rule=prompts.MATH_RULE, language_rule=prompts.language_rule(language)
    )
    return to_text(llm.invoke(prompt)).strip()


def extract_eligibility(llm, store, language="English"):
    docs = multi_query_retrieve(store, prompts.ELIGIBILITY_QUERIES, k_each=4, limit=14)
    prompt = prompts.ELIGIBILITY_PROMPT.format(
        context=format_context(docs),
        not_found=prompts.NOT_FOUND.get(language, prompts.NOT_FOUND["English"]),
        language_rule=prompts.language_rule(language),
    )
    return to_text(llm.invoke(prompt)).strip(), docs


def extract_timeline(llm, store, chunks, language="English"):
    """Dates come from semantic search plus a regex scan of every chunk, so
    dates in tables or annexures are not missed."""
    retrieved = multi_query_retrieve(store, prompts.TIMELINE_QUERIES, k_each=4, limit=10)
    dated = [chunk for chunk in chunks if contains_date(chunk.page_content)]
    docs = dedupe(retrieved + sample_evenly(dated, 10))[:18]
    prompt = prompts.TIMELINE_PROMPT.format(
        context=format_context(docs),
        not_found=prompts.NOT_FOUND.get(language, prompts.NOT_FOUND["English"]),
        language_rule=prompts.language_rule(language),
    )
    return to_text(llm.invoke(prompt)).strip(), docs


def generate_quiz(llm, chunks, language="English", count=5, max_chars=12000):
    """MCQs from chunks sampled across the whole document (not just page 1)."""
    sampled = sample_evenly(chunks, 16)
    texts = take_until([format_context([c]) for c in sampled], max_chars)
    prompt = prompts.QUIZ_PROMPT.format(
        count=count,
        context="\n\n".join(texts),
        math_rule=prompts.MATH_RULE,
        language_rule=prompts.language_rule(language),
    )
    return to_text(llm.invoke(prompt)).strip()
