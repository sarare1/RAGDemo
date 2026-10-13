"""Simple local RAG demo: Streamlit + Ollama + FAISS. Everything runs on your machine."""

import io

import faiss
import numpy as np
import ollama
import streamlit as st
from docx import Document
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Step 1: read the file and return one big string
# ---------------------------------------------------------------------------
def extract_text(file_bytes, filename):
    """Extract plain text from a PDF, DOCX or TXT file."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    # Anything else is treated as a text file
    return file_bytes.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Step 2: split the text into overlapping chunks
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size, overlap):
    """Slide a window of chunk_size characters over the text, stepping by chunk_size - overlap."""
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()  # strip whitespace
        if chunk:  # skip empty chunks
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Step 3: turn text into vectors with Ollama
# ---------------------------------------------------------------------------
def embed_text(text, embed_model):
    """Embed one piece of text; returns a float32 numpy vector."""
    response = ollama.embeddings(model=embed_model, prompt=text)
    return np.array(response["embedding"], dtype="float32")


def embed_chunks(chunks, embed_model, on_progress=None):
    """Embed every chunk; returns a float32 array of shape (n_chunks, dim)."""
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(embed_text(chunk, embed_model))
        if on_progress:
            on_progress((i + 1) / len(chunks))
    return np.array(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# Step 4: store the vectors in a FAISS index
# ---------------------------------------------------------------------------
def build_faiss_index(vectors):
    """Normalize vectors so inner product equals cosine similarity, then index them."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# Step 5: find the chunks most similar to the question
# ---------------------------------------------------------------------------
def retrieve_chunks(question, embed_model, index, chunks, top_k):
    """Embed the question and return the top_k most similar chunks with scores."""
    query = embed_text(question, embed_model).reshape(1, -1)
    faiss.normalize_L2(query)
    scores, positions = index.search(query, top_k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:  # FAISS returns -1 when there are fewer chunks than top_k
            continue
        results.append({"text": chunks[pos], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Step 6: ask the chat model, grounded in the retrieved chunks
# ---------------------------------------------------------------------------
def generate_answer(question, retrieved_chunks, llm_model, temperature):
    """Send the retrieved context and the question to the chat model."""
    context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say \"I don't know based on this document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = ollama.chat(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Helper: turn Ollama errors into friendly messages
# ---------------------------------------------------------------------------
def show_ollama_error(error):
    """Show a clear fix message for common Ollama problems."""
    text = str(error).lower()
    if isinstance(error, ollama.ResponseError) and (
        error.status_code == 404 or "not found" in text
    ):
        st.error(f"Model not found. Run `ollama pull <model>` in a terminal. Details: {error}")
    elif "connect" in text or isinstance(error, ConnectionError):
        st.error("Cannot reach Ollama. Start it with `ollama serve` (or open the Ollama app).")
    else:
        st.error(f"Ollama error: {error}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Local RAG Demo", page_icon="📄")
st.title("📄 Local RAG Demo")

# Sidebar settings
st.sidebar.header("Settings")
embed_model = st.sidebar.text_input("Embedding model", "nomic-embed-text")
llm_model = st.sidebar.text_input("Chat model", "llama3.2")
chunk_size = st.sidebar.slider("Chunk size", 200, 2000, 500, step=100)
overlap = st.sidebar.slider("Overlap", 0, 400, 50, step=25)
top_k = st.sidebar.slider("Top-k", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

# Remember things between reruns
for key in ("chunks", "index", "doc_name", "embed_model_used"):
    st.session_state.setdefault(key, None)

# --- Section 1: upload and process ---
st.subheader("1. Upload a document")
uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])

if st.button("Process Document"):
    if uploaded is None:
        st.warning("Please choose a file first.")
    else:
        try:
            text = extract_text(uploaded.getvalue(), uploaded.name)
            if not text.strip():
                st.warning("No text found in this file.")
            else:
                chunks = chunk_text(text, chunk_size, overlap)
                progress = st.progress(0.0, text="Embedding chunks...")
                vectors = embed_chunks(chunks, embed_model, on_progress=progress.progress)
                st.session_state.index = build_faiss_index(vectors)
                st.session_state.chunks = chunks
                st.session_state.doc_name = uploaded.name
                st.session_state.embed_model_used = embed_model
                st.success(f"Done: {len(chunks)} chunks indexed")
        except Exception as e:
            show_ollama_error(e)

# --- Section 2: ask a question ---
st.subheader("2. Ask a question")
question = st.text_input("Your question")

if st.button("Ask"):
    if st.session_state.index is None:
        st.warning("Please process a document first.")
    elif not question.strip():
        st.warning("Please type a question.")
    elif embed_model != st.session_state.embed_model_used:
        st.warning(
            f"The document was processed with '{st.session_state.embed_model_used}', "
            f"but the embedding model is now '{embed_model}'. Please re-process the document."
        )
    else:
        try:
            with st.spinner("Thinking..."):
                retrieved = retrieve_chunks(
                    question, embed_model, st.session_state.index,
                    st.session_state.chunks, top_k,
                )
                answer = generate_answer(question, retrieved, llm_model, temperature)
            st.markdown("### Answer")
            st.write(answer)
            with st.expander(f"Retrieved chunks from {st.session_state.doc_name}"):
                for i, c in enumerate(retrieved, 1):
                    st.markdown(f"**Chunk {i}** — similarity {c['score']:.3f}")
                    st.write(c["text"])
        except Exception as e:
            show_ollama_error(e)
