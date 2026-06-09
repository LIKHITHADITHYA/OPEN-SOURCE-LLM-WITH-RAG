# NVIDIA Llama-3 Hybrid RAG (Structured)

A professional, modular implementation of a hybrid Retrieval-Augmented Generation (RAG) system using NVIDIA Llama-3 and SerpApi.

## Architecture
- **`src/nvidia_rag/core`**: Main RAG engine, Query Router, and Conversational Memory.
- **`src/nvidia_rag/tools`**: External tools (Web Search via SerpApi, Local Vector DB via ChromaDB).
- **`src/nvidia_rag/ui`**: Advanced Gradio interface with system controls.
- **`src/nvidia_rag/api`**: FastAPI REST server for headless integration.
- **`src/nvidia_rag/config`**: Centralized configuration.

## Features
- **Intelligent Routing:** Automatically decides between web search, local documents, or general knowledge.
- **Persistent Memory:** Remembers conversation history for multi-turn dialogues.
- **Hybrid Retrieval:** Combines live internet data with local document context.
- **API First:** Deployable as a high-performance REST API.

## Usage
### Web UI (Default)
```bash
nvidia-rag
```

### CLI Mode
```bash
nvidia-rag --cli
```

### REST API
```bash
nvidia-rag --api
```

## Development
- Add new tools in `src/nvidia_rag/tools/`.
- Modify LLM behavior in `src/nvidia_rag/core/engine.py`.
- Adjust settings in `src/nvidia_rag/config/settings.py`.
