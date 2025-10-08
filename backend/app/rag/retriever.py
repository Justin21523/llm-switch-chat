"""Document retriever for RAG."""

# backend/app/rag/retriever.py
import logging
from typing import List, Dict, Any, Optional
from .embedder import Embedder
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Document retriever combining embedder and vector store."""

    def __init__(
        self,
        store_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.store_path = store_path
        self.embedding_model = embedding_model

        # Initialize components
        self.embedder = Embedder(embedding_model)
        self.vector_store = VectorStore(store_path, self.embedder.dimension)

    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to the retriever."""
        if not texts:
            return

        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embedder.encode(texts)

            # Add to vector store
            self.vector_store.add_documents(texts, embeddings, metadata)

            # Save to disk
            self.vector_store.save()

            logger.info(f"Successfully added {len(texts)} documents")

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode_single(query)

            # Search vector store
            results = self.vector_store.search(query_embedding, top_k)

            logger.info(f"Retrieved {len(results)} documents for query")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def clear(self):
        """Clear all documents."""
        self.vector_store.clear()
        self.vector_store.save()
        logger.info("Retriever cleared")

    def size(self) -> int:
        """Get number of documents."""
        return self.vector_store.size()
