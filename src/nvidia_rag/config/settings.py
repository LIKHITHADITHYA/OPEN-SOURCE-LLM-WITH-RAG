import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    """System-wide configuration settings."""
    
    # API Keys
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
    
    # Model Configuration
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    TEMPERATURE: float = 0.6
    MAX_TOKENS: int = 1024
    
    # RAG Settings
    SEARCH_CACHE_SIZE: int = 128

settings = Settings()
