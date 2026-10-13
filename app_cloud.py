"""Cloud RAG demo: Streamlit + Google Gemini API + FAISS. For Streamlit Community Cloud."""

import io
import os

import faiss
import numpy as np
import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


# ---------------------------------------------------------------------------
# Gemini client (API key comes from Streamlit secrets or an environment variable)
# ---------------------------------------------------------------------------
def get_client():
    """Create a Gemini client, or return None if no API key is configured."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        key = os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else None


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
    return file_bytes.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Step 2: split the text into overlapping chunks
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size, overlap):
    """Slide a window of chunk_size characters over the text, stepping by chunk_size - overlap."""
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Step 3: turn text into vectors with Gemini
# ---------------------------------------------------------------------------
def embed_text(client, text, embed_model, task_type):
    """Embed one piece of text; returns a float32 numpy vector."""
    result = client.models.embed_content(
        model=embed_model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return np.array(result.embeddings[0].values, dtype="float32")


def embed_chunks(client, chunks, embed_model, on_progress=None):
    """Embed every chunk; returns a float32 array of shape (n_chunks, dim)."""
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(embed_text(client, chunk, embed_model, "RETRIEVAL_DOCUMENT"))
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
def retrieve_chunks(client, question, embed_model, index, chunks, top_k):
    """Embed the question and return the top_k most similar chunks with scores."""
    query = embed_text(client, question, embed_model, "RETRIEVAL_QUERY").reshape(1, -1)
    faiss.normalize_L2(query)
    scores, positions = index.search(query, top_k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue
        results.append({"text": chunks[pos], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# Step 6: ask the chat model, grounded in the retrieved chunks
# ---------------------------------------------------------------------------
def generate_answer(client, question, retrieved_chunks, llm_model, temperature):
    """Send the retrieved context and the question to the chat model."""
    context = "\n\n---\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say \"I don't know based on this document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    response = client.models.generate_content(
        model=llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def show_api_error(error):
    """Show a clear message for common Gemini API problems."""
    text = str(error).lower()
    if "api key" in text or "permission" in text or "401" in text or "403" in text:
        st.error("The Gemini API key was rejected. Check the GEMINI_API_KEY secret.")
    elif "429" in text or "quota" in text:
        st.error("Gemini rate limit reached. Wait a minute and try again.")
    elif "not found" in text or "404" in text:
        st.error(f"Model not found. Check the model name in Settings. Details: {error}")
    else:
        st.error(f"Gemini error: {error}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cloud RAG Demo", page_icon="📄")
st.title("📄 Cloud RAG Demo")

st.sidebar.header("Settings")
embed_model = st.sidebar.text_input("Embedding model", "gemini-embedding-001")
llm_model = st.sidebar.text_input("Chat model", "gemini-2.5-flash")
chunk_size = st.sidebar.slider("Chunk size", 200, 2000, 500, step=100)
overlap = st.sidebar.slider("Overlap", 0, 400, 50, step=25)
top_k = st.sidebar.slider("Top-k", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, step=0.1)

for key in ("chunks", "index", "doc_name", "embed_model_used"):
    st.session_state.setdefault(key, None)

client = get_client()
if client is None:
    st.error("No API key found. Add GEMINI_API_KEY in the app's Secrets settings.")
    st.stop()

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
                vectors = embed_chunks(client, chunks, embed_model, on_progress=progress.progress)
                st.session_state.index = build_faiss_index(vectors)
                st.session_state.chunks = chunks
                st.session_state.doc_name = uploaded.name
                st.session_state.embed_model_used = embed_model
                st.success(f"Done: {len(chunks)} chunks indexed")
        except Exception as e:
            show_api_error(e)

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
                    client, question, embed_model, st.session_state.index,
                    st.session_state.chunks, top_k,
                )
                answer = generate_answer(client, question, retrieved, llm_model, temperature)
            st.markdown("### Answer")
            st.write(answer)
            with st.expander(f"Retrieved chunks from {st.session_state.doc_name}"):
                for i, c in enumerate(retrieved, 1):
                    st.markdown(f"**Chunk {i}** — similarity {c['score']:.3f}")
                    st.write(c["text"])
        except Exception as e:
            show_api_error(e)
