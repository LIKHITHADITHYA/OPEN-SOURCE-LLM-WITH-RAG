"""
Query routing logic.
Uses the LLM to analyze user intent and select the appropriate retrieval tool.
"""

import logging
from typing import Literal
from openai import OpenAI
from nvidia_rag.config.settings import settings

logger = logging.getLogger(__name__)

# Valid routes for the system
RouteType = Literal["WEB", "LOCAL", "NONE"]


class QueryRouter:
    """
    Intelligently routes queries to the best available information source.
    Acts as a 'brain' to prevent unnecessary tool calls.
    """

    def __init__(self, client: OpenAI):
        """
        Initialize the router with an LLM client.

        Args:
            client: An initialized OpenAI/NVIDIA client.
        """
        self.client = client

    def route_query(self, query: str) -> RouteType:
        """
        Determines the optimal route for a given query.

        Args:
            query: The raw user query string.

        Returns:
            'WEB' for external search, 'LOCAL' for internal docs,
            or 'NONE' for general chat.
        """
        # Define the classification prompt for the LLM
        prompt = (
            "You are a routing expert. Classify the user query into "
            "one of three categories:\n"
            "1. 'WEB': For queries about current events, news, or general "
            "world knowledge not likely in local files.\n"
            "2. 'LOCAL': For queries that sound like they refer to internal "
            "documents, manuals, or specific technical specs.\n"
            "3. 'NONE': For greetings, general conversation, or simple "
            "logic that doesn't need external context.\n\n"
            f"Query: {query}\n\n"
            "Return ONLY the word: WEB, LOCAL, or NONE."
        )

        try:
            # Synchronous call to classify the intent
            completion = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic output
                max_tokens=10
            )

            decision = completion.choices[0].message.content.strip().upper()

            # Validation and fallback logic
            if decision in ["WEB", "LOCAL", "NONE"]:
                return decision

            logger.warning(
                "Unexpected routing decision: %s. Defaulting to WEB.",
                decision
            )
            return "WEB"

        except Exception as e:  # noqa: BLE001
            logger.error("Routing failed: %s. Falling back to WEB.", e)
            return "WEB"
