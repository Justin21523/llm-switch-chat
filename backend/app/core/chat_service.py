"""Chat service business logic."""

# backend/app/core/chat_service.py
import logging
from typing import List, Optional, Iterator, Dict, Any
from ..schemas.chat import ChatMessage, ChatRequest, ChatResponse, StreamChunk
from ..backends.base import BaseBackend
from ..rag.retriever import Retriever
from .prompt import PromptTemplate

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service handling conversation logic."""

    def __init__(
        self,
        backend: BaseBackend,
        retriever: Optional[Retriever] = None,
        enable_rag: bool = False,
    ):
        self.backend = backend
        self.retriever = retriever
        self.enable_rag = enable_rag

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate chat response."""
        try:
            # Determine if RAG should be used
            use_rag = (
                request.use_rag if request.use_rag is not None else self.enable_rag
            ) and self.retriever is not None

            rag_context = None

            if use_rag:
                # Retrieve relevant documents
                logger.info(f"Using RAG for query: {request.message}")
                context_docs = self.retriever.retrieve(request.message, top_k=5)

                if context_docs:
                    # Format RAG prompt
                    prompt = PromptTemplate.format_rag_prompt(
                        request.message, context_docs, request.system_prompt
                    )
                    rag_context = PromptTemplate.extract_context_summary(context_docs)
                else:
                    # No relevant docs found, use regular chat
                    messages = request.history + [
                        ChatMessage(role="user", content=request.message)
                    ]
                    prompt = PromptTemplate.format_chat_prompt(
                        messages, request.system_prompt
                    )
            else:
                # Regular chat without RAG
                messages = request.history + [
                    ChatMessage(role="user", content=request.message)
                ]
                prompt = PromptTemplate.format_chat_prompt(
                    messages, request.system_prompt
                )

            # Generate response
            response_text = self.backend.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return ChatResponse(response=response_text, rag_context=rag_context)

        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            raise

    def chat_stream(self, request: ChatRequest) -> Iterator[StreamChunk]:
        """Generate streaming chat response."""
        try:
            # Determine if RAG should be used
            use_rag = (
                request.use_rag if request.use_rag is not None else self.enable_rag
            ) and self.retriever is not None

            if use_rag:
                # Retrieve relevant documents
                logger.info(f"Using RAG for streaming query: {request.message}")
                context_docs = self.retriever.retrieve(request.message, top_k=5)

                if context_docs:
                    # Format RAG prompt
                    prompt = PromptTemplate.format_rag_prompt(
                        request.message, context_docs, request.system_prompt
                    )
                else:
                    # No relevant docs found, use regular chat
                    messages = request.history + [
                        ChatMessage(role="user", content=request.message)
                    ]
                    prompt = PromptTemplate.format_chat_prompt(
                        messages, request.system_prompt
                    )
            else:
                # Regular chat without RAG
                messages = request.history + [
                    ChatMessage(role="user", content=request.message)
                ]
                prompt = PromptTemplate.format_chat_prompt(
                    messages, request.system_prompt
                )

            # Generate streaming response
            for token in self.backend.generate_stream(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield StreamChunk(token=token, finished=False)

            # Send final chunk
            yield StreamChunk(token="", finished=True)

        except Exception as e:
            logger.error(f"Streaming chat generation failed: {e}")
            raise

    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to RAG retriever."""
        if self.retriever is None:
            raise RuntimeError("RAG not enabled - no retriever available")

        self.retriever.add_documents(texts, metadata)
        logger.info(f"Added {len(texts)} documents to RAG system")

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        if self.retriever is None:
            raise RuntimeError("RAG not enabled - no retriever available")

        return self.retriever.retrieve(query, top_k)

    def is_rag_available(self) -> bool:
        """Check if RAG is available."""
        return self.retriever is not None

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if self.retriever is None:
            return {"available": False}

        return {
            "available": True,
            "document_count": self.retriever.size(),
            "embedding_model": self.retriever.embedding_model,
        }
