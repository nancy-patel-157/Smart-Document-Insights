"""Smart Document Insights: a Streamlit RAG app for PDFs.

Run with:  streamlit run app.py
"""

import html
import os

import streamlit as st
from dotenv import load_dotenv

import rag_engine
from helpers import file_fingerprint, friendly_error, source_label
from ui_styles import CSS

load_dotenv()

st.set_page_config(
    page_title="Smart Document Insights",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = ["What is this document about?", "List the key points", "Explain it in simple words"]
LANGUAGE_LABELS = {"English": "English", "Hindi": "हिन्दी"}


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def load_embeddings():
    return rag_engine.get_embeddings()


def empty_state(icon, title, text):
    st.markdown(
        f'<div class="empty"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def section_header(title, subtitle):
    st.markdown(
        f'<div class="section-title">{title}</div><div class="section-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def set_pending(question):
    st.session_state.pending_prompt = question


def to_source_list(documents, preview_chars=300):
    return [
        {"label": source_label(doc.metadata), "text": doc.page_content[:preview_chars].strip() + "..."}
        for doc in documents
    ]


def render_sources(sources, title="Sources used"):
    if not sources:
        return
    with st.expander(title):
        for src in sources:
            st.caption(src["label"])
            st.text(src["text"])


def have_key(api_key):
    if not api_key:
        st.error("Add your Gemini API key in the panel above (or set GOOGLE_API_KEY in .env).")
        return False
    return True


def call_llm(fn, *args, **kwargs):
    """Run an LLM feature and show a friendly error instead of crashing the app."""
    try:
        return fn(*args, **kwargs)
    except Exception as error:  # network, quota, bad key, model errors
        st.error(friendly_error(error))
        return None


def reset_outputs():
    st.session_state.messages = []
    st.session_state.results = {}


def index_documents(uploaded_files, chunk_size, chunk_overlap):
    files = [(f.name, f.getvalue()) for f in uploaded_files]
    fingerprint = file_fingerprint([data for _, data in files], chunk_size, chunk_overlap)
    if fingerprint == st.session_state.fingerprint:
        st.info("These documents are already indexed with the same settings.")
        return

    with st.spinner("Reading and indexing your documents..."):
        documents, page_count, warnings = rag_engine.load_pdfs(files)
        for warning in warnings:
            st.warning(warning)
        if not documents:
            st.error("No readable text found. Scanned PDFs need OCR before they can be indexed.")
            return

        chunks = rag_engine.chunk_documents(documents, chunk_size, chunk_overlap)
        try:
            st.session_state.vector_store = rag_engine.build_index(chunks, load_embeddings())
        except Exception as error:  # e.g. embedding model could not be downloaded
            st.error(f"Indexing failed: {error}")
            return

    text = rag_engine.full_text(documents)
    words = len(text.split())
    st.session_state.chunks = chunks
    st.session_state.full_text = text
    st.session_state.fingerprint = fingerprint
    st.session_state.doc_stats = {
        "pages": page_count,
        "chunks": len(chunks),
        "words": words,
        "minutes": max(1, round(words / 200)),
        "files": ", ".join(name for name, _ in files),
    }
    reset_outputs()
    st.success(f"Indexed {len(files)} file(s) into {len(chunks)} searchable chunks.")


def feature_tab(key, title, subtitle, button_label, empty, runner, filename, api_key, language):
    """Shared layout for Summary / Eligibility / Timeline / Quiz tabs."""
    section_header(title, subtitle)
    if st.session_state.vector_store is None:
        empty_state(*empty)
        return

    gen_col, _ = st.columns([1, 3])
    if gen_col.button(button_label, type="primary", key=f"run_{key}") and have_key(api_key):
        with st.spinner("Working on it..."):
            output = call_llm(runner)
        if output is not None:
            content, docs = output if isinstance(output, tuple) else (output, [])
            st.session_state.results[key] = {
                "content": content,
                "sources": to_source_list(docs),
                "language": language,
            }

    result = st.session_state.results.get(key)
    if not result:
        return
    with st.container(border=True):
        st.markdown(result["content"])
    render_sources(result["sources"], "Excerpts used")
    if result["language"] != language:
        st.markdown(
            f'<p class="lang-note">Generated in {result["language"]}. '
            f"Click “{button_label}” again for {language}.</p>",
            unsafe_allow_html=True,
        )
    st.download_button(
        f"Download {title.lower()} (.md)", data=result["content"], file_name=filename,
        mime="text/markdown", key=f"dl_{key}",
    )


# -------------------------------------------------------------------- app
def main():
    defaults = {
        "messages": [], "vector_store": None, "chunks": [], "full_text": "",
        "doc_stats": {}, "fingerprint": None, "pending_prompt": None, "results": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    nav = st.empty()  # filled at the end so the status is always current

    st.markdown(
        """<div class="hero">
            <h1>Talk to your PDFs.</h1>
            <p>Upload a policy, report, paper or notes. Ask questions with page citations,
            get a 10-point summary, eligibility rules, key deadlines and a practice quiz,
            in English or Hindi.</p>
            <div class="steps">
                <div class="step"><i>1</i>Upload PDFs</div>
                <div class="step"><i>2</i>Index them</div>
                <div class="step"><i>3</i>Ask, summarise, extract or quiz</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ---------------- Upload panel
    api_key = os.getenv("GOOGLE_API_KEY", "")  # from .env, never displayed
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            uploaded_files = st.file_uploader(
                "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
                label_visibility="collapsed",
            )
            if st.session_state.doc_stats:
                files = html.escape(st.session_state.doc_stats["files"])
                st.markdown(f'<div class="files">Indexed: {files}</div>', unsafe_allow_html=True)
        with right:
            st.markdown(
                '<div class="panel-title">Index your document</div>'
                '<div class="panel-text">Indexing splits your PDFs into searchable, page-tagged '
                "pieces. Do this once per upload.</div>",
                unsafe_allow_html=True,
            )
            if not api_key:
                api_key = st.text_input(
                    "Gemini API key", type="password", label_visibility="collapsed",
                    placeholder="Paste your Gemini API key",
                )
            language = st.radio(
                "Answer language", list(LANGUAGE_LABELS), horizontal=True,
                format_func=LANGUAGE_LABELS.get, key="language",
            )
            with st.expander("Advanced settings"):
                chunk_size = st.slider(
                    "Chunk size (characters)", 300, 1200, 1000, 100,
                    help="The embedding model reads ~256 tokens (~1000 characters) per chunk; "
                    "larger chunks would be silently truncated.",
                )
                chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 50)
                top_k = st.slider("Chunks retrieved per question", 2, 8, 4)
            process = st.button("Process and index", type="primary")

    if process:
        if not uploaded_files:
            st.warning("Upload at least one PDF.")
        else:
            index_documents(uploaded_files, chunk_size, chunk_overlap)

    stats = st.session_state.doc_stats
    if stats:
        st.write("")
        tiles = [
            (stats["pages"], "Pages"),
            (f"{stats['words']:,}", "Words"),
            (stats["chunks"], "Searchable chunks"),
            (f"{stats['minutes']} min", "Reading time"),
        ]
        for col, (value, label) in zip(st.columns(4), tiles):
            col.markdown(f'<div class="tile"><b>{value}</b><span>{label}</span></div>', unsafe_allow_html=True)

    st.write("")
    ready = st.session_state.vector_store is not None
    chat_tab, summary_tab, elig_tab, time_tab, quiz_tab = st.tabs(
        ["Chat", "Summary", "Eligibility", "Timeline", "Quiz"]
    )

    # ---------------- Chat
    with chat_tab:
        if st.session_state.messages:
            _, clear_col = st.columns([5, 1])
            if clear_col.button("Clear chat"):
                st.session_state.messages = []
                st.rerun()

        chat_box = st.container()
        typed = st.chat_input(
            "Ask anything about your document..." if ready else "Index a document to start chatting",
            disabled=not ready,
        )
        prompt = typed or st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        with chat_box:
            if not ready:
                empty_state("📂", "No document yet", "Upload a PDF above, then click “Process and index”.")
            elif not st.session_state.messages and not prompt:
                empty_state("💡", "Your document is ready", "Pick a question or type your own below. "
                            "Follow-up questions remember the conversation.")
                for col, q in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
                    col.button(q, key=f"sugg_{q}", on_click=set_pending, args=(q,))

            for message in st.session_state.messages:
                avatar = "🙂" if message["role"] == "user" else "📄"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])
                    render_sources(message.get("sources"))

            if prompt and ready and have_key(api_key):
                history = list(st.session_state.messages)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="🙂"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="📄"):
                    with st.spinner("Searching your document..."):
                        result = call_llm(
                            lambda: rag_engine.answer_question(
                                rag_engine.get_llm(api_key, temperature=0.2),
                                st.session_state.vector_store,
                                prompt,
                                history=history,
                                language=language,
                                k=top_k,
                            )
                        )
                    if result is None:
                        st.session_state.messages.pop()  # let the user retry cleanly
                    else:
                        sources = to_source_list(result["sources"])
                        st.markdown(result["answer"])
                        render_sources(sources)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": result["answer"], "sources": sources}
                        )

    # ---------------- Feature tabs
    def llm():
        return rag_engine.get_llm(api_key, temperature=0.3)

    with summary_tab:
        feature_tab(
            "summary", "Executive summary",
            "Covers the whole document: summary, 10 key points and action items.",
            "Generate summary", ("📝", "Nothing to summarise yet", "Index a document first."),
            lambda: rag_engine.summarize(llm(), st.session_state.full_text, language),
            "Document_Summary.md", api_key, language,
        )
    with elig_tab:
        feature_tab(
            "eligibility", "Eligibility extractor",
            "Who can apply: age limits, qualifications, marks, category rules and required documents.",
            "Extract eligibility", ("🧾", "No document yet", "Index a document first."),
            lambda: rag_engine.extract_eligibility(llm(), st.session_state.vector_store, language),
            "Eligibility.md", api_key, language,
        )
    with time_tab:
        feature_tab(
            "timeline", "Timeline & deadlines",
            "Every date, deadline and milestone found in the document, in order.",
            "Extract timeline", ("📅", "No document yet", "Index a document first."),
            lambda: rag_engine.extract_timeline(
                llm(), st.session_state.vector_store, st.session_state.chunks, language
            ),
            "Timeline.md", api_key, language,
        )
    with quiz_tab:
        feature_tab(
            "quiz", "Revision quiz",
            "Five multiple-choice questions drawn from across the whole document, with an answer key.",
            "Generate 5 questions", ("🎯", "No quiz yet", "Index a document first."),
            lambda: rag_engine.generate_quiz(llm(), st.session_state.chunks, language),
            "Quiz.md", api_key, language,
        )

    st.markdown(
        '<div class="foot">Answers come only from the documents you upload, with page citations.</div>',
        unsafe_allow_html=True,
    )

    status = (
        '<div class="status on"><span class="dot"></span>Document ready</div>'
        if st.session_state.vector_store is not None
        else '<div class="status"><span class="dot"></span>No document yet</div>'
    )
    nav.markdown(
        f'<div class="nav"><div class="nav-brand"><div class="logo">S</div>Smart Document Insights</div>'
        f"{status}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
