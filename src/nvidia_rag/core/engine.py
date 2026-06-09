"""
Main RAG Engine.
Orchestrates the entire generation pipeline: routing, retrieval, memory,
and LLM invocation.
"""

import logging
from typing import Tuple, Optional
from openai import OpenAI
from nvidia_rag.config.settings import settings
from nvidia_rag.tools.search import SearchTool
from nvidia_rag.tools.vector_store import VectorStoreTool
from nvidia_rag.core.router import QueryRouter
from nvidia_rag.core.memory import ConversationMemory

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    The central intelligence of the application.
    Coordinates between tools and models to provide grounded,
    context-aware responses.
    """

    def __init__(self, client: Optional[OpenAI] = None):
        """
        Initialize the RAG Engine with all necessary components.

        Args:
            client: Optional pre-configured OpenAI client. If None,
                    uses default settings.
        """
        # Initialize the API client for NVIDIA NIM
        self.client = client or OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key
        )

        # Initialize retrieval tools
        self.search_tool = SearchTool()
        self.vector_tool = VectorStoreTool()

        # Initialize agentic components
        self.router = QueryRouter(self.client)
        self.memory = ConversationMemory()

    def generate_response(
        self,
        query: str,
        use_rag: bool = True,
        temperature: Optional[float] = None
    ) -> Tuple[str, int, str]:
        """
        Executes the full RAG pipeline for a given user query.

        Args:
            query: The user's input question.
            use_rag: Whether to enable retrieval-based grounding.
            temperature: Optional sampling temperature override.

        Returns:
            A tuple containing (response_text, total_tokens_used,
            source_type_used).
        """
        context = ""
        source_type = "NONE"

        # 1. Step: Intelligent Routing
        # Determine if we need external info or if this is a general chat
        if use_rag:
            source_type = self.router.route_query(query)

            # 2. Step: Context Retrieval
            # Fetch data from the source selected by the router
            if source_type == "WEB":
                context = self.search_tool.search_query(query)
            elif source_type == "LOCAL":
                context = self.vector_tool.search(query)

        # 3. Step: Prompt Construction with Memory
        # Retrieve the dialogue history to maintain multi-turn context
        history = self.memory.get_context()
        system_msg = (
            "You are a helpful AI assistant. Use the provided context to "
            "answer if available."
        )

        # Prepare message list for ChatCompletion
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history)

        # Wrap the user query and retrieved context into the final prompt
        prompt = self._build_rag_prompt(query, context, source_type)
        messages.append({"role": "user", "content": prompt})

        try:
            # 4. Step: LLM Generation
            # Call the NVIDIA-hosted Llama-3 model
            completion = self.client.chat.completions.create(
                model=settings.default_model,
                messages=messages,
                temperature=(
                    temperature
                    if temperature is not None
                    else settings.temperature
                ),
                max_tokens=settings.max_tokens,
                stream=False
            )

            response = completion.choices[0].message.content.strip()
            tokens = completion.usage.total_tokens if completion.usage else 0

            # 5. Step: Memory Update
            # Save the current turn to history for future interactions
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", response)

            return response, tokens, source_type

        except Exception as e:  # noqa: BLE001
            logger.error("LLM generation failed: %s", e)
            return f"Error: {e}", 0, source_type

    def _build_rag_prompt(self, query: str, context: str, source: str) -> str:
        """
        Constructs the final prompt string by combining context and user query.

        Args:
            query: The original user question.
            context: The retrieved text snippet.
            source: The identifier of the retrieval source.

        Returns:
            A formatted prompt string.
        """
        # If no context was found or RAG is disabled, return only the query
        if not context or source == "NONE":
            return query

        return (
            f"Context from {source}:\n"
            f"---------------------\n"
            f"{context}\n"
            f"---------------------\n"
            f"Using the context above and your knowledge, answer "
            f"the following query: {query}"
        )
