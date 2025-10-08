"""Hugging Face Transformers backend."""

# backend/app/backends/transformers_backend.py
import logging
from typing import Iterator, Optional, Dict, Any, List
from ..backends.base import BaseBackend
from ..schemas.chat import ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)


class TransformersBackend(BaseBackend):
    """Hugging Face Transformers backend."""

    def __init__(
        self, model_id: str, device: str = "auto", torch_dtype: str = "auto", **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        self._initialize()

    def _initialize(self):
        """Initialize model and tokenizer."""
        try:

            logger.info(f"Loading model: {self.model_id}")

            # Determine torch dtype
            if self.torch_dtype == "auto":
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                dtype = getattr(torch, self.torch_dtype)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise

    def _load_if_exists(self):
        """Load existing vector store if available."""
        try:
            import faiss

            index_path = self.store_path / "index.faiss"
            data_path = self.store_path / "data.pkl"

            if index_path.exists() and data_path.exists():
                # Load FAISS index
                self._index = faiss.read_index(str(index_path))

                # Load documents and metadata
                with open(data_path, "rb") as f:
                    data = pickle.load(f)

                self._documents = data["documents"]
                self._metadata = data["metadata"]
                self.dimension = data["dimension"]

                logger.info(
                    f"Loaded existing vector store with {len(self._documents)} documents"
                )

        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}")
            # Continue with empty store

    def clear(self):
        """Clear all documents from the vector store."""
        import faiss

        self._index = faiss.IndexFlatIP(self.dimension)
        self._documents = []
        self._metadata = []
        logger.info("Vector store cleared")

    def size(self) -> int:
        """Get number of documents in the store."""
        return len(self._documents)
