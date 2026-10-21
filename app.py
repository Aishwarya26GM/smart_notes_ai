import hashlib
import os
from pathlib import Path

import chromadb
import ollama
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

APP_TITLE = "Smart Notes AI"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM = "llama3.2:1b"
DB_PATH = str(Path(__file__).parent / "data" / "chroma_db")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")
st.title("📚 Smart Notes AI")
st.write("Upload study PDFs, ask questions, summarize notes, and generate practice quizzes.")
st.caption("Local-first demo using Python, Streamlit, ChromaDB, Sentence Transformers, and Ollama.")

@st.cache_resource(show_spinner="Loading the embedding model (the first run may download it)...")
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def read_pdfs(uploaded_files):
    documents = []
    for uploaded in uploaded_files:
        try:
            reader = PdfReader(uploaded)
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for chunk_number, chunk in enumerate(split_text(page_text), start=1):
                    documents.append({
                        "text": chunk,
                        "source": uploaded.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    })
        except Exception as exc:
            st.warning(f"Could not read {uploaded.name}: {exc}")
    return documents

def file_fingerprint(uploaded_files):
    digest = hashlib.sha256()
    for uploaded in uploaded_files:
        digest.update(uploaded.name.encode("utf-8"))
        digest.update(uploaded.getvalue())
    return digest.hexdigest()

def index_documents(documents):
    if not documents:
        raise ValueError("No readable text was found in the PDF files. Scanned PDFs need OCR first.")
    os.makedirs(DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection("smart_notes")
    except Exception:
        pass
    collection = client.create_collection(
        name="smart_notes",
        metadata={"hnsw:space": "cosine"},
    )
    model = get_embedding_model()
    texts = [item["text"] for item in documents]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(documents))]
    metadatas = [
        {"source": item["source"], "page": int(item["page"]), "chunk": int(item["chunk"])}
        for item in documents
    ]
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )
    return len(documents)

def get_collection():
    if not os.path.isdir(DB_PATH):
        return None
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        return client.get_collection("smart_notes")
    except Exception:
        return None

def retrieve(question, top_k=TOP_K):
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return []
    model = get_embedding_model()
    query_embedding = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for i, text in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        items.append({
            "text": text,
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page", "?"),
            "distance": distance,
        })
    return items

def ollama_generate(prompt, model_name):
    try:
        response = ollama.generate(model=model_name, prompt=prompt)
        return response["response"].strip()
    except Exception as exc:
        raise RuntimeError(
            "Could not contact Ollama or the model is not installed. "
            f"Run `ollama pull {model_name}` and make sure Ollama is running. Details: {exc}"
        ) from exc

def make_context(items):
    return "\n\n".join(
        f"[Source: {item['source']}, page {item['page']}]\n{item['text']}"
        for item in items
    )

def answer_question(question, model_name):
    items = retrieve(question)
    if not items:
        raise ValueError("No indexed PDF content found. Upload PDFs and click **Index PDFs** first.")
    context = make_context(items)
    prompt = f"""You are Smart Notes AI, a careful study assistant.
Answer the question using only the CONTEXT below.
If the context does not contain enough information, say that the uploaded notes do not provide enough information.
Do not invent facts. Use simple, student-friendly language.
At the end, add a short 'Sources' list using the source labels supplied in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
    return ollama_generate(prompt, model_name), items

def get_all_document_text(limit=18000):
    collection = get_collection()
    if collection is None or collection.count() == 0:
        return ""
    results = collection.get(include=["documents", "metadatas"])
    pieces = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        pieces.append(f"[{meta.get('source', 'Unknown')}, page {meta.get('page', '?')}]\n{doc}")
    return "\n\n".join(pieces)[:limit]

def build_summary(model_name, style):
    context = get_all_document_text()
    if not context:
        raise ValueError("Upload PDFs and click **Index PDFs** first.")
    prompt = f"""Create a {style.lower()} summary of the study material below.
Use headings and bullet points. Keep it faithful to the supplied text.
Do not add outside facts. If the text is incomplete, do not guess.

STUDY MATERIAL:
{context}

SUMMARY:"""
    return ollama_generate(prompt, model_name)

def build_quiz(model_name, count, difficulty):
    context = get_all_document_text()
    if not context:
        raise ValueError("Upload PDFs and click **Index PDFs** first.")
    prompt = f"""Create {count} multiple-choice questions at {difficulty} difficulty using only the study material below.
For every question, show four options (A-D), then the correct answer and a one-sentence explanation.
Cover different ideas in the material. Do not use outside facts.
Use this readable format:
1. Question
   A. ...
   B. ...
   C. ...
   D. ...
   Answer: B
   Explanation: ...

STUDY MATERIAL:
{context}

QUIZ:"""
    return ollama_generate(prompt, model_name)

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model", value=DEFAULT_LLM)
    st.caption("Install the selected model in a terminal with: `ollama pull " + model_name + "`")
    st.divider()
    st.subheader("1. Upload your notes")
    uploads = st.file_uploader(
        "Choose one or more text-based PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button("Index PDFs", type="primary", use_container_width=True):
        if not uploads:
            st.error("Choose at least one PDF first.")
        else:
            with st.spinner("Reading PDFs, creating embeddings, and saving vectors..."):
                try:
                    docs = read_pdfs(uploads)
                    count = index_documents(docs)
                    st.session_state["indexed_fingerprint"] = file_fingerprint(uploads)
                    st.success(f"Indexed {count} text chunks.")
                except Exception as exc:
                    st.error(str(exc))
    collection = get_collection()
    if collection is not None:
        st.info(f"Indexed chunks available: {collection.count()}")
    else:
        st.info("No indexed documents yet.")

tab_chat, tab_summary, tab_quiz = st.tabs(["💬 Ask your notes", "📝 Summarize", "🧠 Generate quiz"])

with tab_chat:
    st.subheader("Ask a question")
    question = st.text_input("Your question", placeholder="Example: Explain embeddings in simple words")
    if st.button("Get answer", key="answer_btn"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Searching your notes and generating an answer..."):
                try:
                    answer, sources = answer_question(question.strip(), model_name)
                    st.markdown(answer)
                    with st.expander("View retrieved passages"):
                        for i, item in enumerate(sources, start=1):
                            st.markdown(f"**{i}. {item['source']} — page {item['page']}**")
                            st.write(item["text"])
                            st.caption(f"Vector distance: {item['distance']:.3f} (lower is closer)")
                except Exception as exc:
                    st.error(str(exc))

with tab_summary:
    st.subheader("Summarize indexed notes")
    style = st.selectbox("Summary style", ["Short revision summary", "Detailed chapter summary", "Key points for an exam"])
    if st.button("Generate summary", key="summary_btn"):
        with st.spinner("Generating summary with Ollama..."):
            try:
                st.markdown(build_summary(model_name, style))
            except Exception as exc:
                st.error(str(exc))

with tab_quiz:
    st.subheader("Create a practice quiz")
    col1, col2 = st.columns(2)
    with col1:
        count = st.slider("Number of questions", min_value=3, max_value=10, value=5)
    with col2:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    if st.button("Generate quiz", key="quiz_btn"):
        with st.spinner("Creating quiz from your indexed notes..."):
            try:
                st.markdown(build_quiz(model_name, count, difficulty))
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.caption("Privacy note: your PDFs and vector database are processed on your computer. Ollama runs the language model locally; the embedding model is downloaded once when first used.")
