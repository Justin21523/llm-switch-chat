"""OpenAI-compatible backend."""

# backend/app/backends/openai_backend.py
import logging
from typing import Iterator, Optional, Dict, Any, List
from openai import OpenAI
from ..backends.base import BaseBackend
from ..schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class OpenAIBackend(BaseBackend):
    """OpenAI-compatible backend."""

    def __init__(
        self,
        model_id: str,
        base_url: Optional[str] = None,
        api_key: str = "sk-default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self._initialize()

    def _initialize(self):
        """Initialize OpenAI client."""
        try:
            logger.info(f"Initializing OpenAI client for model: {self.model_id}")

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

            logger.info("OpenAI client initialized successfully")

        except ImportError:
            logger.error(
                "openai package not installed. Install with: pip install openai"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate response from prompt."""
        if not self.is_available():
            raise RuntimeError("Client not available")

        try:
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens or 1024,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
                **kwargs,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate streaming response from prompt."""
        if not self.is_available():
            raise RuntimeError("Client not available")

        try:
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            stream = self._client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens or 1024,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
                stream=True,
                **kwargs,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._client is not None

    def generate_with_messages(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate response from chat messages."""
        if not self.is_available():
            raise RuntimeError("Client not available")

        try:
            # Convert to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_messages.append({"role": msg.role, "content": msg.content})

            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=openai_messages,
                max_tokens=max_tokens or 1024,
                temperature=temperature or 0.7,
                top_p=top_p or 0.9,
                **kwargs,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Generation with messages failed: {e}")
            raise
