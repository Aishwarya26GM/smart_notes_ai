# 🧠 Smart Notes AI — RAG-Based Study Assistant

An AI-powered study assistant that helps students understand their notes by uploading PDF documents, asking questions, generating summaries, and creating practice quizzes. Built using **Retrieval-Augmented Generation (RAG)**, semantic search, vector embeddings, and a locally running Large Language Model (LLM).

## 📌 Project Overview

Smart Notes AI transforms static study materials into an interactive learning experience. Instead of manually searching through lengthy PDFs, users can upload their notes and ask questions in natural language.

The system retrieves relevant information from the uploaded documents and uses an AI model to generate context-aware responses. It also supports document summarization and quiz generation to make revision easier.

## ✨ Features

- **📄 PDF Upload:** Upload one or more PDF documents.
- **🔍 Semantic Search:** Retrieve relevant passages based on the meaning of a question.
- **💬 Ask Your Notes:** Ask questions about uploaded study materials and receive AI-generated answers.
- **📚 Source References:** Display the source document and page information for retrieved passages.
- **📝 Summarization:** Generate summaries to understand important concepts quickly.
- **❓ Quiz Generation:** Create practice questions from your study materials.
- **🧠 Local AI Inference:** Use Ollama to run an LLM locally without requiring a paid AI API key.
- **💾 Persistent Vector Storage:** Store document embeddings locally using ChromaDB.
- **🖥️ Simple Web Interface:** Interact with the application through a Streamlit GUI.

## 🏗️ System Architecture

```text
              ┌──────────────────────┐
              │      User / Student  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   Streamlit Web UI   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │     PDF Upload       │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Text Extraction     │
              │       PyPDF           │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Text Chunking        │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Sentence Transformers│
              │  Text Embeddings     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  ChromaDB Vector DB  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Semantic Retrieval   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Ollama + LLM        │
              │  Answer Generation   │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Answer to the User   │
              └──────────────────────┘
```

## ⚙️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Application logic and RAG pipeline |
| Frontend | Streamlit | Interactive web interface |
| PDF Processing | PyPDF | Extract text from PDF documents |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Convert text into vector embeddings |
| Vector Database | ChromaDB | Store and retrieve document embeddings |
| LLM Runtime | Ollama | Run the language model locally |
| Language Model | `llama3.2:1b` | Generate answers, summaries, and quizzes |
| Environment | Python virtual environment | Manage project dependencies |

## 🔄 How It Works

### 1. Document Upload
The user uploads PDF notes through the Streamlit interface.

### 2. Text Extraction
PyPDF extracts text from each page of the uploaded documents.

### 3. Text Chunking
The extracted text is divided into smaller overlapping chunks so that relevant passages can be retrieved efficiently.

### 4. Embedding Generation
Sentence Transformers converts each chunk into a numerical vector representation.

### 5. Vector Storage
ChromaDB stores the embeddings along with the corresponding text and source metadata.

### 6. Semantic Retrieval
When the user asks a question, the system converts the query into an embedding and retrieves the most relevant document chunks.

### 7. Response Generation
The retrieved context is passed to the locally running LLM through Ollama to generate a relevant answer.

### 8. Learning Assistance
Users can also generate summaries and practice quizzes based on the indexed study materials.

## 📂 Project Structure

```text
smart_notes_ai/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── RUN.md                 # Setup and execution instructions
├── EXPLANATION.md          # Project explanation
├── .gitignore              # Files excluded from Git
│
├── .venv/                  # Local Python environment (not committed)
└── data/
    └── chroma_db/          # Persistent vector database (local data)
```

> Note: The `data/chroma_db/` directory is created or populated when the application indexes documents. It may not be present in a fresh repository.

## 🚀 Installation and Setup

### Prerequisites

Install the following before running the project:

- [Python](https://www.python.org/downloads/) — Python 3.11 is a recommended starting point.
- [Git](https://git-scm.com/downloads) — for cloning the repository.
- [Ollama](https://ollama.com/download) — for running the local language model.
- [Visual Studio Code](https://code.visualstudio.com/) — recommended editor.

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/smart_notes_ai.git
cd smart_notes_ai
```

If you already have the project locally, open its folder in VS Code instead.

### Step 2: Create a Virtual Environment

**Windows — Git Bash:**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**Windows — PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download the Ollama Model

Install Ollama and run:

```bash
ollama pull llama3.2:1b
```

The model only needs to be downloaded once unless you remove it.

### Step 5: Start the Application

From the project directory, with the virtual environment activated, run:

```bash
python -m streamlit run app.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

## 💻 Usage

1. Launch the application.
2. Upload one or more PDF study materials through the sidebar.
3. Click **Index PDFs** to extract, chunk, embed, and store the document content.
4. Open the **Ask your notes** tab.
5. Enter a question related to the uploaded documents.
6. Review the generated answer and available source references.
7. Use the summarization and quiz-generation features for revision.

## 🔐 Privacy and Data Storage

- The language model runs locally through Ollama.
- Uploaded document text and vector embeddings are processed and stored locally by the application.
- No paid external LLM API key is required for the configured workflow.
- Python packages and model files may need to be downloaded during initial setup.

Local execution does not automatically guarantee complete privacy for every component. Review the application's dependencies and configuration before using sensitive documents.

## ⚠️ Limitations

- Scanned PDFs containing only images may require OCR before their text can be retrieved.
- Response quality depends on the extracted text, retrieved context, and selected language model.
- Local inference speed depends on available RAM, CPU, and GPU support.
- Very long documents may require careful context selection for summaries and quizzes.
- AI-generated answers and quizzes should be checked against the original study material.
- Re-indexing documents may change the contents of the local vector collection.

## 🔮 Future Enhancements

- Support for DOCX, TXT, and PPTX documents.
- OCR for scanned PDFs.
- Improved source citations with direct page navigation.
- Chat history and saved conversations.
- User authentication and individual document libraries.
- Export summaries and quizzes to PDF.
- More advanced retrieval strategies and reranking.
- Support for multiple locally available LLMs.
- Automated evaluation of retrieval accuracy and answer quality.

