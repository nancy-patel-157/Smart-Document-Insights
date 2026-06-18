import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


st.title("Smart Document Insights")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # -------------------------
    # PDF Reading
    # -------------------------
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # -------------------------
    # Chunking
    # -------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.success(
        f"Total Chunks Created: {len(chunks)}"
    )

    # -------------------------
    # Embeddings
    # -------------------------
    with st.spinner("Creating Embeddings..."):

        model = load_model()

        embeddings = model.encode(chunks)

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

    # -------------------------
    # FAISS Index
    # -------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success(
        f"FAISS Index Created with {index.ntotal} chunks"
    )

    # -------------------------
    # Preview Chunks
    # -------------------------
    if len(chunks) > 0:
        st.subheader("First Chunk")
        st.write(chunks[0])

    if len(chunks) > 1:
        st.subheader("Second Chunk")
        st.write(chunks[1])

    # -------------------------
    # Question Answering
    # -------------------------
    st.divider()

    st.header("Ask Questions From PDF")

    question = st.text_input(
        "Enter your question"
    )

    if question:

        query_embedding = model.encode(
            [question]
        )

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        D, I = index.search(
            query_embedding,
            k=3
        )

        st.subheader("Most Relevant Chunks")

        for rank, idx in enumerate(I[0], start=1):

            st.markdown(
                f"### Result {rank}"
            )

            st.write(chunks[idx])

            st.write("---")