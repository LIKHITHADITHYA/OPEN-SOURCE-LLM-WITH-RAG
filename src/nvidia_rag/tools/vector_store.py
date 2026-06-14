"""
Local document retrieval.
Uses ChromaDB to store and retrieve private document context.
"""

import logging
from typing import List, Optional, Any
from openai import OpenAI

import chromadb

from nvidia_rag.config.settings import settings

logger = logging.getLogger(__name__)


class NVIDIAEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom embedding function to use NVIDIA cloud embeddings."""
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = settings.default_embedding_model

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(input=input, model=self.model)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error("Failed to generate cloud embeddings: %s", e)
            raise


class VectorStoreTool:
    """
    Tool for managing and searching local document embeddings via ChromaDB.
    Enables RAG on private data without external upload.
    """

    def __init__(
        self,
        persist_directory: str = settings.chroma_persist_dir,
        client: Optional[OpenAI] = None
    ):
        """
        Initialize the persistent ChromaDB client.

        Args:
            persist_directory: Folder path for database files.
            client: Shared OpenAI/NVIDIA API client.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use NVIDIA cloud embeddings if client is provided, otherwise fallback to default OpenAI client
        openai_client = client or OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key
        )
        self.embedding_fn = NVIDIAEmbeddingFunction(openai_client)
        
        # Get or create the default collection for the application
        self.collection = self.client.get_or_create_collection(
            name="local_docs",
            embedding_function=self.embedding_fn
        )

    def is_document_ingested(self, file_hash: str) -> bool:
        """
        Checks if a document with the given file hash is already ingested.

        Args:
            file_hash: The SHA-256 hash of the document.

        Returns:
            True if the document exists in the collection, False otherwise.
        """
        try:
            results = self.collection.get(
                where={"doc_hash": file_hash},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to check if document is ingested: %s", e)
            return False

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Ingests new documents into the vector store.

        Args:
            texts: List of document strings to index.
            metadatas: Optional metadata associated with each document.
            ids: Optional unique identifiers. Generated automatically
                 if missing.
        """
        if ids is None:
            ids = [f"id_{i}" for i in range(len(texts))]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(
            "Successfully added %d documents to vector store.", len(texts)
        )

    def search(self, query: str, n_results: int = 3) -> str:
        """
        Retrieves the most semantically relevant documents for a query.

        Args:
            query: The search query string.
            n_results: Number of context fragments to return.

        Returns:
            A formatted string containing the top relevant document snippets.
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

            documents = results.get('documents', [[]])[0]
            if not documents:
                return "No relevant local documents found."

            formatted = "Relevant Local Documents:\n"
            for i, doc in enumerate(documents):
                formatted += f"Source {i+1}:\n{doc}\n\n"
            return formatted

        except Exception as e:  # noqa: BLE001
            logger.error(
                "Vector search failed for query '%s': %s", query, e
            )
            return f"Error retrieving local documents: {e}"

    def get_uploaded_sources(self) -> List[str]:
        """
        Retrieves a list of all unique source filenames currently indexed in the vector store.

        Returns:
            List of strings representing filenames.
        """
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            if not metadatas:
                return []

            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(list(sources))
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to retrieve uploaded sources: %s", e)
            return []
