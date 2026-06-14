"""
Main RAG Engine.
Orchestrates the entire generation pipeline: routing, retrieval, memory,
and LLM invocation.
"""

import logging
import concurrent.futures
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

    def _condense_query(self, query: str) -> str:
        """
        Uses conversation history to rewrite queries with pronouns/references
        into standalone, context-independent questions.
        """
        history = self.memory.get_context()
        if not history:
            return query

        # Format history for the model
        history_str = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = (
            "Given the following conversation history and a follow-up query, "
            "rewrite the follow-up query to be a standalone search query that "
            "contains all necessary context and contains no ambiguous "
            "pronouns/references (like 'it', 'they', 'its', 'he', 'she', "
            "etc.). If the follow-up query is already standalone and does not "
            "need any rewriting, return it exactly as-is.\n\n"
            f"Conversation History:\n{history_str}\n"
            f"Follow-up Query: {query}\n\n"
            "Standalone Query (return ONLY the rewritten query text, no "
            "explanation or formatting):"
        )

        try:
            completion = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128
            )
            content = completion.choices[0].message.content
            if content:
                condensed = content.strip()
                logger.info("Condensed query: '%s' -> '%s'", query, condensed)
                return condensed
            return query
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to condense query: %s", e)
            return query

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
            # First condense the query
            condensed_query = self._condense_query(query)

            uploaded_sources = self.vector_tool.get_uploaded_sources()
            routes = self.router.route_query(condensed_query, uploaded_sources=uploaded_sources)

            # 2. Step: Parallel Context Retrieval
            futures = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if "WEB" in routes:
                    futures["WEB"] = executor.submit(self.search_tool.search_query, condensed_query)
                if "LOCAL" in routes:
                    futures["LOCAL"] = executor.submit(self.vector_tool.search, condensed_query)

            # Gather and validate results
            retrieved_contexts = []
            successful_routes = []

            if "WEB" in futures:
                try:
                    web_context = futures["WEB"].result()
                    # Validate Web Search results: fallback if search failed or returned key error
                    if not (
                        "could not be retrieved" in web_context
                        or "error" in web_context.lower()
                    ):
                        retrieved_contexts.append(f"Web Search Context:\n{web_context}")
                        successful_routes.append("WEB")
                    else:
                        logger.warning("Web search source is invalid or failed.")
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Web search execution failed: %s", e)

            if "LOCAL" in futures:
                try:
                    local_context = futures["LOCAL"].result()
                    # Validate Vector Store results: fallback if no documents matched
                    # or db query errored
                    if not (
                        "error" in local_context.lower()
                        or "no relevant local documents" in local_context.lower()
                    ):
                        retrieved_contexts.append(f"Local Document Context:\n{local_context}")
                        successful_routes.append("LOCAL")
                    else:
                        logger.warning("Local document source is invalid or empty.")
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Local vector search execution failed: %s", e)

            if retrieved_contexts:
                context = "\n\n".join(retrieved_contexts)
                source_type = ", ".join(successful_routes)
            else:
                context = ""
                source_type = "NONE"

        # 3. Step: Prompt Construction with Memory
        # Retrieve the dialogue history to maintain multi-turn context
        history = self.memory.get_context()
        system_msg = (
            "You are a helpful conversational AI assistant. "
            "If the query is a simple greeting or general small talk, respond naturally "
            "and ignore the provided context. Otherwise, use the provided context to "
            "answer if it is relevant to the query."
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

            content = completion.choices[0].message.content
            response = content.strip() if content is not None else ""
            tokens = completion.usage.total_tokens if completion.usage else 0

            # 5. Step: Memory Update
            # Save the current turn to history for future interactions
            self.memory.add_turn("user", query)
            self.memory.add_turn("assistant", response)

            return response, tokens, source_type

        except Exception as e:  # pylint: disable=broad-except
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
