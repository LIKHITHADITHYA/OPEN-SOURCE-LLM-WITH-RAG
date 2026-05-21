import functools
import logging
from typing import List, Dict, Any, Union
from langchain_community.utilities import SerpAPIWrapper
from nvidia_rag.config.settings import settings

logger = logging.getLogger(__name__)

class SearchTool:
    """Wrapper for real-time search capabilities."""
    
    def __init__(self, api_key: str = settings.SERPAPI_API_KEY):
        if not api_key:
            logger.warning("SERPAPI_API_KEY not found. Search functionality will be limited.")
        self.search = SerpAPIWrapper(serpapi_api_key=api_key)

    @functools.lru_cache(maxsize=settings.SEARCH_CACHE_SIZE)
    def search_query(self, query: str) -> str:
        """
        Executes a search and returns a formatted string of results.
        Cached based on settings.
        """
        try:
            results = self.search.run(query)
            return self._format_results(results)
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return "Real-time search results could not be retrieved due to an error."

    def _format_results(self, results: Union[str, List[Dict[str, Any]]]) -> str:
        """Parses and formats raw search results."""
        if isinstance(results, str):
            return results
            
        if not isinstance(results, list):
            return "No search results found."

        formatted = []
        for i, res in enumerate(results):
            title = res.get('title', 'N/A')
            link = res.get('link', 'N/A')
            source = res.get('source', 'N/A')
            date = res.get('date', 'N/A')
            formatted.append(
                f"Result {i+1}:\n"
                f"Title: {title}\n"
                f"Link: {link}\n"
                f"Source: {source}\n"
                f"Date: {date}\n"
            )
        return "\n".join(formatted)
