"""Dependency injection for FastAPI."""

# backend/app/deps.py``
import logging
from functools import lru_cache
from typing import Optional
from .settings import settings
from .backends.registry import registry
from .core.chat_service import ChatService
from .rag.retriever import Retriever

logger = logging.getLogger(__name__)


@lru_cache()
def get_backend():
    """Get LLM backend instance (cached)."""
    try:
        config = {
            "model_id": settings.MODEL_ID,
        }

        if settings.MODEL_BACKEND == "hf":
            config.update(
                {
                    "device": settings.HF_DEVICE,
                    "torch_dtype": settings.HF_TORCH_DTYPE,
                }
            )
        elif settings.MODEL_BACKEND == "llamacpp":
            config.update(
                {
                    "gguf_path": settings.GGUF_PATH,
                    "n_ctx": settings.LLAMACPP_N_CTX,
                    "n_gpu_layers": settings.LLAMACPP_N_GPU_LAYERS,
                }
            )
        elif settings.MODEL_BACKEND == "openai":
            config.update(
                {
                    "base_url": settings.OPENAI_BASE_URL,
                    "api_key": settings.OPENAI_API_KEY,
                }
            )

        backend = registry.get_backend(settings.MODEL_BACKEND, **config)
        logger.info(f"Backend {settings.MODEL_BACKEND} created successfully")
        return backend

    except Exception as e:
        logger.error(f"Failed to create backend: {e}")
        raise


@lru_cache()
def get_retriever() -> Optional[Retriever]:
    """Get RAG retriever instance (cached)."""
    if not settings.ENABLE_RAG:
        return None

    try:
        retriever = Retriever(
            store_path=settings.VECTOR_STORE_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
        )
        logger.info("RAG retriever created successfully")
        return retriever

    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        # Return None instead of raising to allow non-RAG operation
        return None


def get_chat_service() -> ChatService:
    """Get chat service instance."""
    backend = get_backend()
    retriever = get_retriever()

    service = ChatService(
        backend=backend, retriever=retriever, enable_rag=settings.ENABLE_RAG
    )

    return service
