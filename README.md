# Smart Notes AI

A local-first RAG study assistant built with Python and Streamlit.

## Features

- PDF question answering using Retrieval-Augmented Generation
- Semantic search with Sentence Transformers and ChromaDB
- Summaries for revision
- Multiple-choice quiz generation
- Source filename and page metadata
- Local language model inference through Ollama

## Quick start

See [RUN.md](RUN.md) for full setup instructions.

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:1b
python -m streamlit run app.py
```

## Documentation

- [RUN.md](RUN.md): installation, execution, and troubleshooting
- [EXPLANATION.md](EXPLANATION.md): architecture and concept explanations

## Note

The first run downloads the embedding model and Ollama model. Scanned PDFs require OCR, which is not included in this starter version.
