"""
FastAPI Server implementation.
Provides a REST interface for interacting with the RAG engine programmatically.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import uvicorn

from nvidia_rag.core.engine import RAGEngine

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
    Upload a text document to the local vector database.

    Processes the file into chunks and indexes them for RAG.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported for now."
        )

    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Naive chunking
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        engine.vector_tool.add_documents(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": len(chunks)
        }
    except Exception as e:  # noqa: BLE001
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
