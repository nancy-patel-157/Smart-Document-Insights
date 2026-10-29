# Smart Document Insights

Smart Document Insights is a Retrieval-Augmented Generation (RAG) app for PDFs, built for
**Smart India Hackathon (SIH)**-style use cases such as government schemes, policies,
notices and study material. Upload one or more PDFs, index them once, and then chat with
them, summarise them, pull out eligibility rules and deadlines, or generate a revision quiz,
in **English or Hindi**, with page citations.

## Key Features

* **Chat with page citations:** answers are grounded only in the uploaded documents and cite
  the file and page they came from, e.g. `[policy.pdf, p. 4]`. If the answer is not in the
  document, the app says so instead of guessing.
* **Follow-up memory:** questions like "explain that in more detail" are rewritten into a
  standalone question using the recent conversation before searching.
* **Session caching:** the vector index lives in Streamlit session state, so questions never
  re-run the embedding pipeline. Re-processing the same files with the same settings is skipped.
* **10-Point Summary:** executive summary, exactly 10 numbered key points and action items.
  Long documents are summarised section by section (map-reduce), so the whole document is
  covered, not just the first few pages.
* **Eligibility Extractor:** age limits, qualifications, minimum marks, category, income and
  document requirements, as a table with sources.
* **Timeline & Deadline Tracker:** every date and milestone in chronological order, found by
  semantic search *and* a date-pattern scan of every chunk (so dates in tables are not missed).
* **Revision Quiz:** 5 MCQs with an answer key, drawn from chunks spread across the whole document.
* **Hindi support:** one toggle switches chat answers, summaries, extractors and quizzes to Hindi.
* **Math support:** formulas are returned as LaTeX (`$...$`, `$$...$$`) and rendered in the app.
* **Downloads:** every generated report can be downloaded as Markdown.
* **Friendly errors:** invalid API keys, rate limits, scanned or password-protected PDFs are
  reported clearly instead of crashing the app.

## How It Works

```
Indexing (once per upload)
PDF ──► text per page (PyPDF) ──► chunks with {file, page} ──► embeddings (MiniLM) ──► FAISS

Question time
question ──► (rewrite follow-up) ──► top-k similar chunks ──► Gemini prompt with
             cited context ──► answer + sources
```

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Sentence-Transformers `all-MiniLM-L6-v2` (runs locally, free) |
| Vector store | FAISS (in-memory) |
| LLM | Google Gemini (`gemini-2.5-flash` by default, configurable) |

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nancy-patel-157/Smart-Document-Insights.git
   cd Smart-Document-Insights
   ```

2. **Create a virtual environment and install dependencies** (Python 3.10+)
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Add your Gemini API key**
   ```bash
   cp .env.example .env      # Windows: copy .env.example .env
   ```
   Then put your key from [Google AI Studio](https://aistudio.google.com/apikey) in `.env`.
   You can also paste the key into the app instead.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The first run downloads the embedding model (~90 MB), later runs use the cache.

## Usage

1. Upload one or more PDFs and click **Process and index**.
2. Pick **English** or **हिन्दी** for answers.
3. Use the tabs: **Chat**, **Summary**, **Eligibility**, **Timeline**, **Quiz**.
4. Open **Sources used** under any answer to see the exact excerpts and pages.

**Advanced settings:** chunk size (max 1200 characters, because the embedding model only reads
about 256 tokens per chunk), chunk overlap, and how many chunks are retrieved per question.

## Project Structure

```
app.py            Streamlit UI (layout, tabs, session state)
rag_engine.py     PDF loading, indexing, retrieval and all LLM features (no UI code)
prompts.py        All prompt templates and language rules
helpers.py        Small dependency-free utilities (date detection, sampling, errors)
ui_styles.py      CSS theme
tests/            Unit tests with a fake LLM and fake embeddings (no API key needed)
```

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Limitations

* Scanned (image-only) PDFs need OCR first; tables and images are read as plain text only.
* The index is kept in memory per browser session and is rebuilt after a page refresh.
* `all-MiniLM-L6-v2` is English-centric, so retrieval on Hindi-language PDFs is weaker
  (answers can still be written in Hindi).
* Gemini free-tier rate limits apply.
