# 📄 Smart Document Insights

Smart Document Insights is an advanced, production-ready Retrieval-Augmented Generation (RAG) application built for the **Smart India Hackathon (SIH)** use cases. It allows users to upload complex documents, extract structural text seamlessly, perform high-speed semantic searches via a vector database, and extract highly specific policy metadata using the Gemini API.

## 🚀 Key Features (SIH-Level Workflow)
* **⚡ Instant Semantics (Session Caching):** Remembers document status in memory via Streamlit Session State. Submitting a question skips the ingestion/embedding pipeline entirely for immediate execution.
* **💬 Context-Insulated Q&A:** Answers user queries strictly based on the extracted document bounds to eliminate LLM hallucinations.
* **📋 10-Point Policy Summarizer:** Generates an executive numbered analytical summary of core regulations instantly.
* **🎓 Eligibility Extractor:** Automatically parses demographic criteria, academic benchmarks, and age limits.
* **📅 Timeline & Deadline Tracker:** Extracts critical dates, registration schedules, and milestone events into a scannable format.
* **🌐 Hindi Language Support:** Features a global native routing toggle that dynamically switches processing, summaries, and extraction layouts into fluent Hindi.
* **📐 Structural Math Support:** Native system prompts push structural formula outputs into clean LaTeX configurations ($...$ and $$...$$).

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Text Processing:** PyPDF & LangChain Text Splitters
* **Vector Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Vector Database Engine:** FAISS (Facebook AI Similarity Search)
* **LLM Engine:** Google Gemini Pro (`gemini-1.5-flash`)

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/nancy-patel-157/Smart-Document-Insights.git](https://github.com/nancy-patel-157/Smart-Document-Insights.git)
   cd Smart-Document-Insights
