"""
System-wide configuration management.
Handles loading environment variables and setting default parameters for the
LLM and RAG tools.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings container for the RAG application.

    Attributes:
        nvidia_api_key: Secret key for NVIDIA NIM API.
        serpapi_api_key: Secret key for Google search retrieval.
        nvidia_base_url: Endpoint for the NVIDIA OpenAI-compatible API.
        default_model: The specific Llama-3 model to use for generation
                       and routing.
        temperature: Sampling temperature for LLM responses.
        max_tokens: Maximum response length.
        search_cache_size: Number of search results to cache in memory.
        chroma_persist_dir: Local directory for vector database storage.
        memory_turns: Number of previous dialogue turns to retain in context.
    """

    # API Keys
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    serpapi_api_key: str = os.getenv("SERPAPI_API_KEY", "")

    # Model Configuration
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    default_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    temperature: float = 0.6
    max_tokens: int = 1024

    # RAG Settings
    search_cache_size: int = 128
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    memory_turns: int = 5


# Singleton settings instance used across the project
settings = Settings()
