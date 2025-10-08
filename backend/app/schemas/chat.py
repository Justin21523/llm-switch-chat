"""Chat API schemas."""

# backend/app/schemas/chat.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., description="User message")
    history: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt override"
    )
    temperature: Optional[float] = Field(
        default=None, description="Generation temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    use_rag: Optional[bool] = Field(
        default=None, description="Enable RAG for this request"
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str = Field(..., description="Assistant response")
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Token usage info"
    )
    rag_context: Optional[str] = Field(default=None, description="RAG context used")


class StreamChunk(BaseModel):
    """Stream response chunk."""

    token: str = Field(..., description="Generated token")
    finished: bool = Field(default=False, description="Whether generation is finished")
    usage: Optional[Dict[str, Any]] = Field(
        default=None, description="Token usage (only in final chunk)"
    )


class RAGUploadRequest(BaseModel):
    """RAG document upload request."""

    content: str = Field(..., description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Document metadata"
    )


class RAGUploadResponse(BaseModel):
    """RAG document upload response."""

    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="Status message")
    document_id: Optional[str] = Field(default=None, description="Document ID")


class RAGQueryRequest(BaseModel):
    """RAG query request."""

    query: str = Field(..., description="Query text")
    top_k: Optional[int] = Field(default=5, description="Number of results to return")


class RAGQueryResponse(BaseModel):
    """RAG query response."""

    results: List[Dict[str, Any]] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="Original query")
