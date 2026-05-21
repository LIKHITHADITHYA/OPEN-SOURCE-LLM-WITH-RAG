# NVIDIA Llama-3 Hybrid RAG (Structured)

A professional, modular implementation of a hybrid Retrieval-Augmented Generation (RAG) system using NVIDIA Llama-3 and SerpApi.

## Architecture
- **`src/nvidia_rag/core`**: Main RAG engine and LLM logic.
- **`src/nvidia_rag/tools`**: External tool integrations (Search, etc.).
- **`src/nvidia_rag/ui`**: Web interface components (Gradio).
- **`src/nvidia_rag/config`**: Centralized configuration and environment management.

## Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the package:
   ```bash
   pip install -e .
   ```
4. Set up your `.env` file with `NVIDIA_API_KEY` and `SERPAPI_API_KEY`.

## Usage
### Web UI (Default)
Run the application to launch the Gradio interface:
```bash
nvidia-rag
```

### CLI Mode
For a lightweight terminal experience:
```bash
nvidia-rag --cli
```

## Development
- Add new tools in `src/nvidia_rag/tools/`.
- Modify LLM behavior in `src/nvidia_rag/core/engine.py`.
- Adjust settings in `src/nvidia_rag/config/settings.py`.
