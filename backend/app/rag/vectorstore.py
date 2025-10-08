"""Vector store for RAG documents."""
# backend/app/rag/vectorstore.py
import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store."""

    def __init__(self, store_path: str, dimension: int = 384):
        self.store_path = Path(store_path)
        self.dimension = dimension
        self._index = None
        self._documents = []
        self._metadata = []
        self._initialize()

    def _initialize(self):
        """Initialize FAISS index."""
        try:
            # Create index
            self._index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)

            # Load existing data if available
            self._load_if_exists()

            logger.info(f"Vector store initialized with {len(self._documents)} documents")

        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu or faiss-gpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]] = None
    ):
        """Add documents to the vector store."""
        if embeddings.shape[0] != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension must be {self.dimension}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        self._index.add(embeddings.astype(np.float32))

        # Store documents and metadata
        self._documents.extend(texts)
        if metadata is None:
            metadata = [{"id": len(self._documents) + i} for i in range(len(texts))]
        self._metadata.extend(metadata)

        logger.info(f"Added {len(texts)} documents to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if len(self._documents) == 0:
            return []

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self._index.search(query_embedding, min(top_k, len(self._documents)))

        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append({
                    "text": self._documents[idx],
                    "score": float(score),
                    "metadata": self._metadata[idx]
                })

        return results

    def save(self):
        """Save vector store to disk."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = self.store_path / "index.faiss"
            faiss.write_index(self._index, str(index_path))

            # Save documents and metadata
            data = {
                "documents": self._documents,
                "metadata": self._metadata,
                "dimension": self.dimension
            }

            data_path = self.store_path / "data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Vector store saved to {self.store_path}")

        except Exception as e:

