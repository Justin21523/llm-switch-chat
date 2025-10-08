"""Prompt templates and formatting."""

# backend/app/core/prompt.py
import logging
from typing import List, Optional, Dict, Any
from ..schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Prompt template manager."""

    DEFAULT_SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. You provide accurate and helpful responses to user questions."""

    RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Use the provided context information to answer the user's question accurately. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Please answer the following question based on the provided context:"""

    @classmethod
    def format_chat_prompt(
        cls, messages: List[ChatMessage], system_prompt: Optional[str] = None
    ) -> str:
        """Format chat messages into a prompt."""
        # Use provided system prompt or default
        if system_prompt is None:
            system_prompt = cls.DEFAULT_SYSTEM_PROMPT

        # Start with system message
        formatted_parts = [f"System: {system_prompt}"]

        # Add conversation history
        for msg in messages:
            if msg.role == "user":
                formatted_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted_parts.append(f"Assistant: {msg.content}")

        # Add final assistant prompt
        formatted_parts.append("Assistant:")

        return "\n\n".join(formatted_parts)

    @classmethod
    def format_rag_prompt(
        cls,
        query: str,
        context_docs: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format RAG prompt with context."""
        # Build context string
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[{i}] {doc['text']}")

        context_str = "\n\n".join(context_parts)

        # Use RAG system prompt or provided one
        if system_prompt is None:
            system_prompt = cls.RAG_SYSTEM_PROMPT.format(context=context_str)
        else:
            system_prompt = f"{system_prompt}\n\nContext:\n{context_str}"

        # Format as chat
        messages = [ChatMessage(role="user", content=query)]
        return cls.format_chat_prompt(messages, system_prompt)

    @classmethod
    def extract_context_summary(cls, context_docs: List[Dict[str, Any]]) -> str:
        """Extract a summary of the context for response metadata."""
        if not context_docs:
            return ""

        summaries = []
        for doc in context_docs[:3]:  # Top 3 documents
            text = doc["text"]
            # Truncate long texts
            if len(text) > 200:
                text = text[:200] + "..."
            summaries.append(f"- {text}")

        return "\n".join(summaries)
