"""
FastAPI Server implementation.
Provides a REST interface for interacting with the RAG engine programmatically.
"""

import hashlib
import io
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pypdf  # pylint: disable=import-error
import uvicorn

from nvidia_rag.core.engine import RAGEngine
from nvidia_rag.tools.splitter import split_text_recursively

# Initialize the FastAPI application
app = FastAPI(
    title="NVIDIA Llama-3 Hybrid RAG API",
    description=(
        "A high-performance RAG backend providing web and "
        "local document grounding."
    ),
    version="1.0.0"
)

# Initialize the shared engine instance
engine = RAGEngine()


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint serving a premium dark-themed documentation landing page.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NVIDIA Llama-3 Hybrid RAG API</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-gradient: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --accent-green: #76b900;
                --accent-green-glow: rgba(118, 185, 0, 0.4);
                --card-bg: rgba(31, 41, 55, 0.6);
                --card-border: rgba(75, 85, 99, 0.4);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg-gradient);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                overflow-x: hidden;
                padding: 2rem;
            }
            .container {
                max-width: 800px;
                width: 100%;
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--card-border);
                border-radius: 24px;
                padding: 3rem;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5), 0 0 50px rgba(118, 185, 0, 0.1);
                text-align: center;
                animation: fadeIn 0.8s ease-out;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .logo-container {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 10px;
                margin-bottom: 1.5rem;
            }
            .logo-icon {
                font-size: 2.5rem;
                animation: pulse 2s infinite alternate;
            }
            @keyframes pulse {
                0% { transform: scale(1); filter: drop-shadow(0 0 2px var(--accent-green)); }
                100% { transform: scale(1.1); filter: drop-shadow(0 0 12px var(--accent-green)); }
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                letter-spacing: -0.5px;
            }
            h1 span {
                color: var(--accent-green);
                text-shadow: 0 0 15px var(--accent-green-glow);
            }
            .subtitle {
                font-size: 1.1rem;
                color: var(--text-secondary);
                margin-bottom: 2.5rem;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }
            .endpoints { text-align: left; margin-bottom: 2.5rem; }
            .endpoints-title {
                font-size: 1.2rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: var(--text-primary);
                border-bottom: 1px solid var(--card-border);
                padding-bottom: 0.5rem;
            }
            .endpoint-card {
                background: rgba(17, 24, 39, 0.5);
                border: 1px solid rgba(75, 85, 99, 0.2);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .endpoint-card:hover {
                border-color: var(--accent-green);
                transform: translateX(5px);
                box-shadow: 0 0 15px rgba(118, 185, 0, 0.1);
            }
            .endpoint-info { display: flex; align-items: center; gap: 12px; }
            .method {
                font-size: 0.85rem;
                font-weight: 800;
                padding: 0.25rem 0.6rem;
                border-radius: 6px;
                text-transform: uppercase;
            }
            .method.post {
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
                border: 1px solid rgba(16, 185, 129, 0.4);
            }
            .method.get {
                background: rgba(59, 130, 246, 0.2);
                color: #3b82f6;
                border: 1px solid rgba(59, 130, 246, 0.4);
            }
            .path { font-family: monospace; font-size: 1.05rem; font-weight: 600; color: #f3f4f6; }
            .desc { font-size: 0.9rem; color: var(--text-secondary); }
            .actions { display: flex; justify-content: center; gap: 15px; }
            .btn {
                font-family: 'Outfit', sans-serif;
                font-size: 1rem;
                font-weight: 600;
                padding: 0.8rem 2rem;
                border-radius: 12px;
                cursor: pointer;
                text-decoration: none;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
            }
            .btn-primary {
                background: var(--accent-green);
                color: #0b0f19;
                border: none;
                box-shadow: 0 4px 14px var(--accent-green-glow);
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(118, 185, 0, 0.6);
            }
            .btn-secondary {
                background: transparent;
                color: var(--text-primary);
                border: 1px solid var(--card-border);
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.05);
                border-color: var(--text-primary);
                transform: translateY(-2px);
            }
            footer { margin-top: 3rem; font-size: 0.85rem; color: #4b5563; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo-container">
                <span class="logo-icon">🟢</span>
            </div>
            <h1>NVIDIA Llama-3 <span>Hybrid RAG API</span></h1>
            <p class="subtitle">A high-performance REST backend providing grounded agentic generation using NVIDIA NIMs, SerpApi, and ChromaDB.</p>
            
            <div class="endpoints">
                <h2 class="endpoints-title">Available Endpoints</h2>
                
                <div class="endpoint-card">
                    <div class="endpoint-info">
                        <span class="method post">post</span>
                        <span class="path">/chat</span>
                    </div>
                    <span class="desc">Query the hybrid RAG system</span>
                </div>

                <div class="endpoint-card">
                    <div class="endpoint-info">
                        <span class="method post">post</span>
                        <span class="path">/upload</span>
                    </div>
                    <span class="desc">Ingest text/PDF documents</span>
                </div>

                <div class="endpoint-card">
                    <div class="endpoint-info">
                        <span class="method get">get</span>
                        <span class="path">/health</span>
                    </div>
                    <span class="desc">Check service health status</span>
                </div>
            </div>

            <div class="actions">
                <a href="/docs" class="btn btn-primary">Open Interactive Swagger Docs</a>
                <a href="/health" class="btn btn-secondary">Check Health</a>
            </div>
        </div>
        <footer>
            NVIDIA Llama-3 REST API Server v1.0.0
        </footer>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


class QueryRequest(BaseModel):
    """Schema for incoming chat requests."""
    query: str
    use_rag: bool = True
    temperature: Optional[float] = None


class QueryResponse(BaseModel):
    """Schema for outgoing chat responses."""
    response: str
    tokens: int
    source: str


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """
    Primary endpoint for generating grounded responses.

    Expects a JSON payload with 'query' and optional 'use_rag'
    and 'temperature' flags.
    """
    try:
        # Pass request to the core engine
        resp, tokens, source = engine.generate_response(
            request.query,
            request.use_rag,
            request.temperature
        )
        return QueryResponse(response=resp, tokens=tokens, source=source)
    except Exception as e:
        # Standard error response
        raise HTTPException(
            status_code=500,
            detail=f"Engine error: {str(e)}"
        ) from e


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a text or PDF document to the local vector database.

    Processes the file into chunks and indexes them for RAG.
    """
    filename_lower = file.filename.lower()
    if not (filename_lower.endswith(".txt") or filename_lower.endswith(".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported for now."
        )

    try:
        content = await file.read()
        if filename_lower.endswith(".pdf"):
            pdf_file = io.BytesIO(content)
            reader = pypdf.PdfReader(pdf_file)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            text = "\n".join(text_parts)
        else:
            text = content.decode("utf-8")

        if not text.strip():
            raise ValueError("The uploaded file has no readable text content.")

        file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if engine.vector_tool.is_document_ingested(file_hash):
            return {
                "status": "skipped",
                "filename": file.filename,
                "detail": "Document already ingested. Skipping."
            }

        chunks = split_text_recursively(text, chunk_size=1000, chunk_overlap=200)
        if not chunks:
            raise ValueError("The split text generated no chunks.")

        metadatas = [{"source": file.filename, "doc_hash": file_hash} for _ in chunks]
        ids = [
            f"{file.filename}_{file_hash}_{i}_{uuid.uuid4().hex[:8]}"
            for i in range(len(chunks))
        ]
        engine.vector_tool.add_documents(chunks, metadatas=metadatas, ids=ids)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}"
        ) from e


@app.get("/health")
async def health():
    """Simple health check endpoint for monitoring."""
    return {"status": "healthy", "model": "Llama-3"}


def main():
    """Entry point for the uvicorn server."""
    # Run the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
