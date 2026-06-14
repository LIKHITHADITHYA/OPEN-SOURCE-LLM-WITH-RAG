"""
Query routing logic.
Uses the LLM to analyze user intent and select the appropriate retrieval tool.
"""

import logging
from typing import Literal, List, Optional
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

    def route_query(
        self,
        query: str,
        uploaded_sources: Optional[List[str]] = None
    ) -> List[RouteType]:
        """
        Determines the optimal route(s) for a given query.

        Args:
            query: The raw user query string.
            uploaded_sources: A list of unique filenames currently uploaded.

        Returns:
            A list of RouteType (e.g. ['WEB'], ['LOCAL'], ['WEB', 'LOCAL'], or ['NONE']).
        """
        # Format the uploaded sources to let the LLM know what is available
        sources_str = ", ".join(uploaded_sources) if uploaded_sources else "None"

        # Define the classification prompt for the LLM
        prompt = (
            "You are a routing expert. Classify the user query into "
            "one or more of three categories:\n"
            "1. 'WEB': For queries about current events, news, or general "
            "world knowledge not likely in local files.\n"
            "2. 'LOCAL': For queries that sound like they refer to internal "
            "documents, manuals, or specific technical specs. "
            f"Currently uploaded documents in the database are: {sources_str}.\n"
            "3. 'NONE': For greetings, general conversation, or simple "
            "logic that doesn't need external context.\n\n"
            "If the query requires information from multiple sources, choose all "
            "that apply and separate them with a comma (e.g., 'WEB, LOCAL').\n\n"
            f"Query: {query}\n\n"
            "Return ONLY the comma-separated category names (e.g., "
            "WEB or LOCAL or WEB, LOCAL or NONE)."
        )

        try:
            # Synchronous call to classify the intent
            completion = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic output
                max_tokens=10
            )

            content = completion.choices[0].message.content
            if content is None:
                return ["WEB"]

            decisions = [x.strip().upper() for x in content.split(",")]
            valid_decisions = [d for d in decisions if d in ["WEB", "LOCAL", "NONE"]]

            # Filter out NONE if other valid sources are selected
            if "NONE" in valid_decisions and len(valid_decisions) > 1:
                valid_decisions = [d for d in valid_decisions if d != "NONE"]

            if valid_decisions:
                return valid_decisions

            logger.warning(
                "Unexpected routing decision: %s. Defaulting to ['WEB'].",
                content
            )
            return ["WEB"]

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Routing failed: %s. Falling back to ['WEB'].", e)
            return ["WEB"]
