# NVIDIA Llama-3 Hybrid RAG (Structured)

A professional, modular implementation of a hybrid Retrieval-Augmented Generation (RAG) system using NVIDIA Llama-3.3 (via NIM endpoints), SerpApi, and a local ChromaDB vector database.

---

## Key Features
- **Intelligent Document-Aware Routing:** Automatically routes queries between `LOCAL` documents, `WEB` search, or `NONE` (general conversation). The LLM is dynamically fed the names of uploaded documents to make highly precise routing decisions.
- **Hybrid Retrieval:** Integrates real-time internet search results (via SerpApi) with local vector database matches (via ChromaDB).
- **Multi-Format Ingestion:** Supports ingestion of both plain text (`.txt`) and PDF (`.pdf`) documents. Chunks, embeds (using the local `all-MiniLM-L6-v2` model), and stores documents persistently on disk.
- **Source Validation & Fallback:** Automatically intercepts retrieval errors (such as missing search keys or empty databases) and gracefully falls back to a general knowledge response (`NONE` route), avoiding crashing or returning error logs to users.
- **Stateful Dialogue Memory:** Maintains a rolling conversation history buffer to preserve context during multi-turn dialogue sessions.
- **Flexible UI/CLI Modes:** Launch the system as an interactive Gradio web application or a lightweight terminal CLI.

---

## Project Directory Structure

```text
Nvidia-Llama-RAG/
├── src/
│   └── nvidia_rag/
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py       # Global settings container & environment loader
│       ├── core/
│       │   ├── __init__.py
│       │   ├── engine.py         # Main RAG orchestration pipeline & validation
│       │   ├── memory.py         # Stateful rolling conversation memory buffer
│       │   └── router.py         # LLM-based query router (WEB vs LOCAL vs NONE)
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── search.py         # Google Search via SerpApi with LRU caching
│       │   └── vector_store.py   # ChromaDB persistent client, indexing, & file listing
│       ├── ui/
│       │   ├── __init__.py
│       │   └── web_ui.py         # Gradio Chat Interface and document upload panel
│       ├── __init__.py
│       └── main.py               # Unified application launcher (Web UI, CLI)
├── tests/
│   ├── test_engine.py            # Live integration test suite (requires API keys)
│   └── test_unit.py              # Mock-based unit test suite (offline verification)
├── .env.example                  # Template configuration file
├── pyproject.toml                # Build system definitions and project metadata
├── requirements.txt              # Project package dependencies list
└── README.md                     # Documentation
```

---

## Setup & Configuration

### 1. Set Up Environment Variables
Create a `.env` file in the project root directory and define the following variables:

```env
# NVIDIA NIM API Key (Required for Llama-3.3 LLM operations)
NVIDIA_API_KEY=your_nvidia_api_key_here

# SerpApi API Key (Required for live Google Search)
SERPAPI_API_KEY=your_serpapi_key_here

# Local directory where ChromaDB stores document vectors (Defaults to ./chroma_db)
CHROMA_PERSIST_DIR=./chroma_db
```

### 2. Install Project Dependencies
Use the project's virtual environment to install the package dependencies:
```bash
test_venv/bin/pip install -r requirements.txt
```

---

## Running the Application

Always configure the Python path when launching the modules directly:

### 1. Launch the Gradio Web UI (Default)
Launches the browser interface on `http://127.0.0.1:7860` for chatting and uploading files:
```bash
PYTHONPATH=src test_venv/bin/python -m nvidia_rag.main
```

### 2. Launch the Terminal CLI
Launches a clean, interactive dialogue shell directly in your console:
```bash
PYTHONPATH=src test_venv/bin/python -m nvidia_rag.main --cli
```

---

## Running Tests

### 1. Run Unit Tests (Offline Validation)
Executes **16 mocked unit tests** to verify memory sliding, document chunking, prompt templating, and error fallback states:
```bash
PYTHONPATH=src test_venv/bin/python tests/test_unit.py
```

### 2. Run Integration Tests (Online Validation)
Tests live API endpoints. Requires valid API keys set in your `.env` file:
```bash
PYTHONPATH=src test_venv/bin/python tests/test_engine.py
```
