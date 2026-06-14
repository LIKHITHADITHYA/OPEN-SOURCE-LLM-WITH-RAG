"""
FastAPI Server implementation.
Provides a REST interface for interacting with the RAG engine programmatically.
"""

import hashlib
import io
from typing import Optional
import uuid

from fastapi import FastAPI, HTTPException, File, UploadFile
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
