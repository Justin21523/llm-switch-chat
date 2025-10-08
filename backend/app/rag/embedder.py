"""Embedding model for RAG."""

# backend/app/rag/embedder.py
import logging
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Text embedder using sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._initialize()

    def _initialize(self):
        """Initialize embedding model."""
        try:

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not self.is_available():
            raise RuntimeError("Embedding model not available")

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        return self.encode([text])[0]

    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if not self.is_available():
            raise RuntimeError("Embedding model not available")
        return self._model.get_sentence_embedding_dimension()
