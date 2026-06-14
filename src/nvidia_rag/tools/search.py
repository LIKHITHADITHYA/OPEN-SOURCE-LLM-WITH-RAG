"""
Web search utility.
Interfaces with SerpApi to retrieve real-time internet context for RAG.
"""

import functools
import logging
from typing import List, Dict, Any, Union
from langchain_community.utilities import SerpAPIWrapper
from nvidia_rag.config.settings import settings

logger = logging.getLogger(__name__)


class SearchTool:
    """
    Wrapper for real-time web search capabilities using SerpApi.
    Provides cached results to minimize API costs and latency.
    """

    def __init__(self, api_key: str = settings.serpapi_api_key):
        """
        Initialize the SerpAPI wrapper.

        Args:
            api_key: The SerpAPI secret key.
        """
        if not api_key:
            logger.warning(
                "SERPAPI_API_KEY not found. Search functionality will "
                "be unavailable or limited."
            )

        self.search = SerpAPIWrapper(serpapi_api_key=api_key)

    @functools.lru_cache(maxsize=settings.search_cache_size)
    def search_query(self, query: str) -> str:
        """
        Executes a search and returns a formatted string of the top results.
        Results are cached based on the query string.

        Args:
            query: The search terms.

        Returns:
            A formatted string containing snippets and metadata from
            search results.
        """
        try:
            # Execute the search via LangChain's community utility
            results = self.search.run(query)
            return self._format_results(results)

        except Exception as e:  # noqa: BLE001
            logger.error("Search failed for query '%s': %s", query, e)
            return (
                "Real-time search results could not be retrieved "
                "due to an error."
            )

    def _format_results(
        self,
        results: Union[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Parses and formats raw search results into a clean text snippet.

        Args:
            results: The raw output from SerpApi (can be a string or a list).

        Returns:
            A string ready for inclusion in an LLM prompt.
        """
        # If it's already a string (common for single-snippet results),
        # return it
        if isinstance(results, str):
            return results

        if not isinstance(results, list):
            return "No search results found."

        # Process multiple result objects
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
