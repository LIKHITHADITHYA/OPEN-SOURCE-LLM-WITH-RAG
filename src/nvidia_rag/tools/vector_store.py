"""
Local document retrieval.
Uses ChromaDB to store and retrieve private document context.
"""

import logging
from typing import List, Optional

import chromadb

from nvidia_rag.config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreTool:
    """
    Tool for managing and searching local document embeddings via ChromaDB.
    Enables RAG on private data without external upload.
    """

    def __init__(self, persist_directory: str = settings.chroma_persist_dir):
        """
        Initialize the persistent ChromaDB client.

        Args:
            persist_directory: Folder path for database files.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Get or create the default collection for the application
        self.collection = self.client.get_or_create_collection(
            name="local_docs"
        )

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
