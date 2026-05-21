import logging
from typing import Tuple, Optional
from openai import OpenAI
from nvidia_rag.config.settings import settings
from nvidia_rag.tools.search import SearchTool

logger = logging.getLogger(__name__)

class RAGEngine:
    """Core RAG logic for grounded generation."""
    
    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client or OpenAI(
            base_url=settings.NVIDIA_BASE_URL,
            api_key=settings.NVIDIA_API_KEY
        )
        self.search_tool = SearchTool()

    def generate_response(self, query: str, use_rag: bool = True) -> Tuple[str, int, int]:
        """
        Generates a response using either hybrid RAG or baseline LLM.
        Returns: (text, tokens, search_calls)
        """
        search_results = ""
        search_calls = 0
        
        if use_rag:
            search_results = self.search_tool.search_query(query)
            search_calls = 1 if "Result 1:" in search_results else 0
            prompt = self._build_rag_prompt(query, search_results)
            system_msg = "You are a helpful AI assistant that provides grounded answers."
        else:
            prompt = query
            system_msg = "You are a helpful AI assistant."

        try:
            completion = self.client.chat.completions.create(
                model=settings.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                stream=False
            )
            
            response = completion.choices[0].message.content.strip()
            tokens = completion.usage.total_tokens if completion.usage else 0
            return response, tokens, search_calls
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error: {e}", 0, search_calls

    def _build_rag_prompt(self, query: str, results: str) -> str:
        """Constructs the RAG prompt."""
        if "could not be retrieved" in results or "No search results found" in results:
            return f"I was unable to retrieve external results. Answer based on internal knowledge: {query}"
            
        return (
            f"Based on the following search results and your knowledge, answer the query.\n\n"
            f"Search Results:\n{results}\n\n"
            f"User Query: {query}"
        )
