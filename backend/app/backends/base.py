"""Base backend interface."""

# backend/app/backends/base.py
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Dict, Any, List
from ..schemas.chat import ChatMessage


class BaseBackend(ABC):
    """Base class for LLM backends."""

    def __init__(self, **kwargs):
        """Initialize backend with configuration."""
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate streaming response from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into prompt string."""
        formatted_parts = []
        for msg in messages:
            if msg.role == "system":
                formatted_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")

        return "\n\n".join(formatted_parts)
