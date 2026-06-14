"""
Conversation memory management.
Provides a stateful buffer to track dialogue history for multi-turn
interactions.
"""

from typing import List, Dict
from nvidia_rag.config.settings import settings


class ConversationMemory:
    """
    Manages short-term conversational context using a rolling buffer.
    Ensures the LLM stays within context window limits by pruning old turns.
    """

    def __init__(self, max_turns: int = settings.memory_turns):
        """
        Initialize the memory buffer.

        Args:
            max_turns: Maximum number of user-assistant pairs to keep.
        """
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        """
        Adds a single message to the history.

        Args:
            role: The sender role ('user' or 'assistant').
            content: The message text.
        """
        self.history.append({"role": role, "content": content})

        # Prune history if it exceeds double the turn limit (pairs)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_context(self) -> List[Dict[str, str]]:
        """
        Returns the current conversation history as a list of message objects.

        Returns:
            List of dictionaries compatible with OpenAI's ChatCompletion API.
        """
        return self.history

    def format_for_prompt(self) -> str:
        """
        Converts history into a single string for legacy completion prompts.

        Returns:
            A formatted string representing the dialogue history.
        """
        if not self.history:
            return ""

        formatted = "Previous conversation:\n"
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        return formatted + "\n"

    def clear(self):
        """Resets the history buffer, effectively starting a new session."""
        self.history = []
