"""End-to-end tests of the RAG engine with a fake LLM and fake embeddings.

No API key or model download is needed. Skipped automatically if the
LangChain / FAISS packages are not installed.
"""

import io

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_community")
pytest.importorskip("faiss")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

import rag_engine  # noqa: E402


class FakeLLM:
    """Records prompts and returns canned replies (mimics .invoke / .batch)."""

    def __init__(self, reply="fake answer"):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return type("Msg", (), {"content": self.reply})()

    def batch(self, prompts):
        return [self.invoke(p) for p in prompts]


@pytest.fixture
def pages():
    return [
        Document(page_content="Applicants must be between 18 and 25 years of age.",
                 metadata={"source": "policy.pdf", "page": 1}),
        Document(page_content="The last date of registration is 15/03/2025.",
                 metadata={"source": "policy.pdf", "page": 2}),
        Document(page_content="Minimum 60% marks in Class XII are required.",
                 metadata={"source": "policy.pdf", "page": 3}),
    ]


@pytest.fixture
def store(pages):
    chunks = rag_engine.chunk_documents(pages, chunk_size=300, chunk_overlap=50)
    return rag_engine.build_index(chunks, DeterministicFakeEmbedding(size=32)), chunks


def test_chunks_keep_page_metadata(pages):
    long_page = Document(page_content="word " * 400, metadata={"source": "a.pdf", "page": 7})
    chunks = rag_engine.chunk_documents([long_page], chunk_size=300, chunk_overlap=500)
    assert len(chunks) > 1
    assert all(c.metadata == {"source": "a.pdf", "page": 7} for c in chunks)


def test_answer_includes_page_labels_and_language(store):
    vector_store, _ = store
    llm = FakeLLM("They must be 18-25 [policy.pdf, p. 1]")
    result = rag_engine.answer_question(llm, vector_store, "What is the age limit?", language="Hindi", k=2)
    assert result["answer"].startswith("They must be")
    assert len(result["sources"]) == 2
    prompt = llm.prompts[-1]
    assert "[policy.pdf, p." in prompt
    assert "Hindi" in prompt
    assert len(llm.prompts) == 1  # no history -> no condense call


def test_follow_up_question_is_condensed(store):
    vector_store, _ = store
    llm = FakeLLM("What is the minimum marks requirement?")
    history = [
        {"role": "user", "content": "Tell me about marks"},
        {"role": "assistant", "content": "60% in Class XII"},
    ]
    result = rag_engine.answer_question(llm, vector_store, "explain more", history=history)
    assert len(llm.prompts) == 2
    assert "Tell me about marks" in llm.prompts[0]
    assert result["standalone_question"] == "What is the minimum marks requirement?"


def test_short_summary_is_single_call():
    llm = FakeLLM("summary")
    assert rag_engine.summarize(llm, "short text") == "summary"
    assert len(llm.prompts) == 1
    assert "10 Key Points" in llm.prompts[0]


def test_long_summary_uses_map_reduce():
    llm = FakeLLM("- note")
    text = "\n".join("sentence number %d about the policy." % i for i in range(3000))  # ~110k chars
    rag_engine.summarize(llm, text)
    map_calls = [p for p in llm.prompts if p.startswith("Summarise this part")]
    assert 2 <= len(map_calls) <= 11
    assert "Notes from part 1" in llm.prompts[-1]
    # The end of the document reaches the model, not just the first 12k characters.
    assert any("sentence number 2999" in p for p in map_calls)


def test_timeline_includes_regex_dated_chunks(store):
    vector_store, chunks = store
    llm = FakeLLM("| Date | Event | Source |")
    content, docs = rag_engine.extract_timeline(llm, vector_store, chunks)
    assert content.startswith("| Date")
    assert any("15/03/2025" in d.page_content for d in docs)


def test_eligibility_returns_sources(store):
    vector_store, _ = store
    llm = FakeLLM("| Criterion | Requirement | Source |")
    content, docs = rag_engine.extract_eligibility(llm, vector_store)
    assert docs and len(docs) == len({(d.metadata["page"], d.page_content) for d in docs})


def test_quiz_samples_whole_document():
    chunks = [
        Document(page_content=f"fact {i}", metadata={"source": "a.pdf", "page": i}) for i in range(1, 101)
    ]
    llm = FakeLLM("quiz")
    rag_engine.generate_quiz(llm, chunks)
    prompt = llm.prompts[0]
    assert "a.pdf, p. 1]" in prompt and "a.pdf, p. 94]" in prompt  # first and near-last


def test_to_text_handles_content_blocks():
    msg = type("M", (), {"content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]})()
    assert rag_engine.to_text(msg) == "Hello world"
    assert rag_engine.to_text(type("M", (), {"content": "plain"})()) == "plain"


def test_load_pdfs_reads_pages_and_reports_bad_files():
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    pdf.drawString(72, 720, "Page one text")
    pdf.showPage()
    pdf.showPage()  # blank page
    pdf.drawString(72, 720, "Page three text")
    pdf.save()

    docs, page_count, warnings = rag_engine.load_pdfs(
        [("sample.pdf", buffer.getvalue()), ("broken.pdf", b"not a pdf")]
    )
    assert page_count == 3
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert "Page three" in docs[1].page_content
    assert any("broken.pdf" in w for w in warnings)
    assert any("1 page(s) had no text" in w for w in warnings)
