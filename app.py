import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai
import os
import faiss
import numpy as np


# -------------------------
# Load Gemini API Key
# -------------------------

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# -------------------------
# Load Embedding Model
# -------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# -------------------------
# App Title
# -------------------------

st.title("Smart Document Insights")


# -------------------------
# Upload PDF
# -------------------------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # -------------------------
    # Read PDF
    # -------------------------

    documents = []

    reader = PdfReader(uploaded_file)

    for page_num, page in enumerate(
        reader.pages
    ):

        page_text = page.extract_text()

        if page_text:

            documents.append(
                {
                    "page": page_num + 1,
                    "text": page_text
                }
            )

    # -------------------------
    # Chunking
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunk_data = []

    for doc in documents:

        page_chunks = splitter.split_text(
            doc["text"]
        )

        for chunk in page_chunks:

            if len(chunk.strip()) > 100:

                chunk_data.append(
                    {
                        "chunk": chunk,
                        "page": doc["page"]
                    }
                )

    chunks = [
        item["chunk"]
        for item in chunk_data
    ]

    st.success(
        f"Total Chunks Created: {len(chunks)}"
    )

    # -------------------------
    # Embeddings
    # -------------------------

    with st.spinner(
        "Creating embeddings..."
    ):

        model = load_model()

        embeddings = model.encode(
            chunks
        )

        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

    # -------------------------
    # Create FAISS Index
    # -------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    st.success(
        f"FAISS Index Created with {index.ntotal} chunks"
    )

    # -------------------------
    # Ask Question
    # -------------------------

    st.divider()

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        # -------------------------
        # Query Embedding
        # -------------------------

        query_embedding = model.encode(
            [question]
        )

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        # -------------------------
        # Search FAISS
        # -------------------------

        D, I = index.search(
            query_embedding,
            k=3
        )

        relevant_chunks = []

        for idx in I[0]:

            relevant_chunks.append(
                chunk_data[idx]
            )

        # -------------------------
        # Build Context
        # -------------------------

        context = "\n\n".join(
            [
                f"""
Page: {item['page']}

Content:
{item['chunk']}
                """
                for item in relevant_chunks
            ]
        )

        # -------------------------
        # Gemini Prompt
        # -------------------------

        prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{question}

Rules:
1. Do not make up information.
2. Mention page numbers.
3. If answer not found, say:
   "Answer not found in document."
"""

        # -------------------------
        # Generate Answer
        # -------------------------

        with st.spinner(
            "Generating answer..."
        ):

            try:

                response = gemini.generate_content(
                    prompt
                )

                st.subheader(
                    "Answer"
                )

                if hasattr(
                    response,
                    "text"
                ):

                    st.write(
                        response.text
                    )

                else:

                    st.error(
                        "No response generated"
                    )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

        # -------------------------
        # Show Sources
        # -------------------------

        st.subheader(
            "Sources"
        )

        pages = sorted(
            list(
                set(
                    [
                        item["page"]
                        for item in relevant_chunks
                    ]
                )
            )
        )

        for page in pages:

            st.write(
                f"Page {page}"
            )