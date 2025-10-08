"""Backend registry for model selection."""

# backend/app/backends/registry.py
import logging
from typing import Dict, Type
from .base import BaseBackend
from .transformers_backend import TransformersBackend
from .llamacpp_backend import LlamaCppBackend
from .openai_backend import OpenAIBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """Registry for managing different backends."""

    _backends: Dict[str, Type[BaseBackend]] = {
        "hf": TransformersBackend,
        "transformers": TransformersBackend,
        "llamacpp": LlamaCppBackend,
        "llama.cpp": LlamaCppBackend,
        "openai": OpenAIBackend,
        "openai-compatible": OpenAIBackend,
    }

    @classmethod
    def get_backend(cls, backend_type: str, **config) -> BaseBackend:
        """Get backend instance by type."""
        backend_type = backend_type.lower()

        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(
                f"Unknown backend type: {backend_type}. Available: {available}"
            )

        backend_class = cls._backends[backend_type]

        try:
            logger.info(f"Creating backend: {backend_type}")
            return backend_class(**config)
        except Exception as e:
            logger.error(f"Failed to create backend {backend_type}: {e}")
            raise

    @classmethod
    def register_backend(cls, name: str, backend_class: Type[BaseBackend]):
        """Register a custom backend."""
        cls._backends[name] = backend_class
        logger.info(f"Registered custom backend: {name}")

    @classmethod
    def list_backends(cls) -> list:
        """List available backend types."""
        return list(cls._backends.keys())


# Global registry instance
registry = BackendRegistry()
