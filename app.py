import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Phase 2: Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    st.success(f"Total Chunks Created: {len(chunks)}")

    # Phase 3: Embeddings
    model = load_model()

    embeddings = model.encode(chunks)

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # Phase 3: FAISS
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    st.success(
        f"FAISS Index Created with {index.ntotal} chunks"
    )

    st.subheader("First Chunk")
    st.write(chunks[0])

    if len(chunks) > 1:
        st.subheader("Second Chunk")
        st.write(chunks[1])