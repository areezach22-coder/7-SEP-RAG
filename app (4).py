import os
import tempfile

import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF, build a FAISS vector index, and ask questions using "
    "an open-source model through Groq."
)


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Open-source embedding model from Hugging Face
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# PDF extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n\n".join(pages)


# -----------------------------
# Text chunking
# -----------------------------
def create_chunks(text, chunk_size=800, chunk_overlap=150):
    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


# -----------------------------
# FAISS vector database
# -----------------------------
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


def search_faiss(query, chunks, index, top_k=4):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append(
                {
                    "text": chunks[idx],
                    "score": float(score),
                }
            )

    return results


# -----------------------------
# Groq generation
# -----------------------------
def generate_answer(question, retrieved_chunks, api_key, model_name):
    client = Groq(api_key=api_key)

    context = "\n\n".join(
        [
            f"[Source {i + 1}]\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = """
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.
If the answer is not present in the context, clearly say that the
information was not found in the uploaded document.

Do not invent facts.
Keep the answer clear and concise.
"""

    user_prompt = f"""
Document context:

{context}

Question:
{question}

Answer based only on the document context.
"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="For deployment, you can store this in Streamlit Secrets instead.",
    )

    model_name = st.selectbox(
        "Groq open-source model",
        [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ],
        index=0,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=400,
        max_value=1500,
        value=800,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=50,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=8,
        value=4,
    )


# -----------------------------
# PDF upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"],
)

if uploaded_file is not None:
    if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.pdf_text = None
        st.session_state.chunks = None
        st.session_state.index = None

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Build Vector Database", type="primary"):
        with st.spinner("Extracting PDF, creating chunks, and building FAISS index..."):
            try:
                text = extract_text_from_pdf(uploaded_file)

                if not text.strip():
                    st.error(
                        "No selectable text was found in this PDF. "
                        "Scanned/image-only PDFs need OCR before this app can read them."
                    )
                    st.stop()

                chunks = create_chunks(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                index, _ = create_faiss_index(chunks)

                st.session_state.pdf_text = text
                st.session_state.chunks = chunks
                st.session_state.index = index

                st.success(
                    f"Vector database created successfully. "
                    f"{len(chunks)} chunks were indexed."
                )

            except Exception as e:
                st.error(f"Error while processing the PDF: {e}")


# -----------------------------
# Document information
# -----------------------------
if st.session_state.get("chunks"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Characters", f"{len(st.session_state.pdf_text):,}")

    with col2:
        st.metric("Chunks", len(st.session_state.chunks))

    with col3:
        st.metric("Embedding dimension", 384)

    with st.expander("Preview extracted text"):
        st.text(st.session_state.pdf_text[:5000])


# -----------------------------
# Question answering
# -----------------------------
st.divider()
st.subheader("💬 Ask a question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What are the main conclusions of this document?",
)

if st.button("Ask Question"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif st.session_state.get("index") is None:
        st.warning("Please click 'Build Vector Database' first.")

    else:
        # Read API key from the input first.
        api_key = groq_api_key

        # If no key was entered, try Streamlit Secrets.
        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            st.error(
                "Groq API key is missing. Enter it in the sidebar or add "
                "GROQ_API_KEY to Streamlit Secrets."
            )

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching the document and generating an answer..."):
                try:
                    results = search_faiss(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        top_k=top_k,
                    )

                    answer = generate_answer(
                        question,
                        results,
                        api_key,
                        model_name,
                    )

                    st.subheader("Answer")
                    st.write(answer)

                    st.subheader("Retrieved Sources")

                    for i, result in enumerate(results):
                        with st.expander(
                            f"Source {i + 1} | Similarity: {result['score']:.3f}"
                        ):
                            st.write(result["text"])

                except Exception as e:
                    st.error(f"Error while answering the question: {e}")
else:
    st.info("Upload a PDF and build the vector database to start asking questions.")
