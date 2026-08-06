import os
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA

load_dotenv()

# Page Layout & Config
st.set_page_config(
    page_title="Smart Document Insights Pro",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern Light UI Styling
st.markdown("""
    <style>
    /* Main App Clean Light Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }

    /* Title & Headers */
    h1, h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Modern Card Layout for Stats */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 26px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 6px;
        color: #475569;
        font-weight: 600;
        border: none;
        padding: 0px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #2563eb !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Action Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_with_metadata(pdf_docs):
    text = ""
    page_count = 0
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        page_count += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text, page_count

def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def main():
    st.title("📄 Smart Document Insights Pro")
    st.caption("Contextual Document Intelligence & RAG Query Engine")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "extracted_text" not in st.session_state:
        st.session_state.extracted_text = ""
    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = {}

    # Sidebar Controls
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        
        st.subheader("RAG Parameters")
        chunk_size = st.slider("Chunk Size", 300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        
        if st.button(" Process & Index"):
            if not api_key:
                st.error("Provide a Gemini API Key.")
                return
            if not uploaded_files:
                st.warning("Upload at least one PDF.")
                return

            with st.spinner("Indexing document vectors..."):
                raw_text, page_count = extract_text_with_metadata(uploaded_files)
                if not raw_text.strip():
                    st.error("No extractable text found.")
                    return

                chunks = get_text_chunks(raw_text, chunk_size, chunk_overlap)
                vector_store = get_vector_store(chunks)

                st.session_state.extracted_text = raw_text
                st.session_state.vector_store = vector_store
                st.session_state.doc_stats = {
                    "pages": page_count,
                    "chunks": len(chunks),
                    "words": len(raw_text.split())
                }
                st.success("Indexing Complete!")

        # Analytics Overview in Sidebar
        if st.session_state.doc_stats:
            st.divider()
            st.markdown("### 📊 Document Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{st.session_state.doc_stats['pages']}</div>
                        <div class="metric-label">Pages</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{st.session_state.doc_stats['chunks']}</div>
                        <div class="metric-label">Chunks</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.doc_stats['words']}</div>
                    <div class="metric-label">Total Words</div>
                </div>
            """, unsafe_allow_html=True)

    # Main Workspace
    tab1, tab2, tab3 = st.tabs(["💬 Interactive Chat   ", "📝 Executive Summary    ", "🎯 Quiz Generator"])

    with tab1:
        st.subheader("Chat with Your Document")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask anything about your uploaded documents..."):
            if not st.session_state.vector_store:
                st.info("Please upload and index documents first using the sidebar.")
                return

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching document context..."):
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 4}),
                        return_source_documents=True
                    )
                    res = qa_chain.invoke({"query": prompt})
                    answer = res["result"]
                    sources = res["source_documents"]

                    st.markdown(answer)
                    
                    with st.expander("📌 View Source Context Chunks"):
                        for idx, doc in enumerate(sources):
                            st.caption(f"**Chunk {idx+1}:**")
                            st.text(doc.page_content[:300] + "...")

            st.session_state.messages.append({"role": "assistant", "content": answer})

    with tab2:
        st.subheader("Executive Analysis")
        if st.button("Generate Comprehensive Report"):
            if not st.session_state.extracted_text:
                st.info("Upload documents first.")
            else:
                with st.spinner("Analyzing document structure..."):
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                    prompt = f"""
                    Analyze the following text and format as Markdown:
                    # Executive Summary
                    Provide 3-4 clear summary paragraphs.
                    
                    # Key Takeaways
                    - Bullet points of top technical/business insights.
                    
                    # Action Items & Decisions
                    - Recommended next steps based on document content.
                    
                    Content: {st.session_state.extracted_text[:12000]}
                    """
                    response = llm.invoke(prompt)
                    report = response.content
                    st.markdown(report)
                    st.download_button("📥 Download Analysis (.md)", data=report, file_name="Document_Analysis.md", mime="text/markdown")

    with tab3:
        st.subheader("Automated Review Tools")
        if st.button("Generate 5 Revision MCQs"):
            if not st.session_state.extracted_text:
                st.info("Upload documents first.")
            else:
                with st.spinner("Generating quiz..."):
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                    quiz_prompt = f"Create 5 multiple choice questions with answer keys based on this text:\n{st.session_state.extracted_text[:10000]}"
                    quiz_response = llm.invoke(quiz_prompt)
                    st.markdown(quiz_response.content)

if __name__ == "__main__":
    main()