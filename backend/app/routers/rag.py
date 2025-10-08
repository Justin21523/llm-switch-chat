"""RAG (Retrieval-Augmented Generation) endpoints."""

# backend/app/routers/rag.py
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List, Dict, Any
import io

from ..schemas.chat import (
    RAGUploadRequest,
    RAGUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
)
from ..core.chat_service import ChatService
from ..deps import get_chat_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/upload", response_model=RAGUploadResponse)
async def upload_document(
    request: RAGUploadRequest, service: ChatService = Depends(get_chat_service)
) -> RAGUploadResponse:
    """Upload document to RAG system."""
    try:
        if not service.is_rag_available():
            raise HTTPException(status_code=400, detail="RAG not enabled")

        # Split content into chunks (simple approach)
        chunks = _split_text(request.content)

        # Prepare metadata
        metadata = []
        base_metadata = request.metadata or {}
        for i, chunk in enumerate(chunks):
            chunk_metadata = {**base_metadata, "chunk_id": i}
            metadata.append(chunk_metadata)

        # Add to RAG system
        service.add_documents(chunks, metadata)

        return RAGUploadResponse(
            success=True,
            message=f"Successfully uploaded document with {len(chunks)} chunks",
        )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-file", response_model=RAGUploadResponse)
async def upload_file(
    file: UploadFile = File(...), service: ChatService = Depends(get_chat_service)
) -> RAGUploadResponse:
    """Upload file to RAG system."""
    try:
        if not service.is_rag_available():
            raise HTTPException(status_code=400, detail="RAG not enabled")

        # Read file content
        content_bytes = await file.read()

        # Parse content based on file type
        if file.filename.endswith(".txt"):
            content = content_bytes.decode("utf-8")
        elif file.filename.endswith(".md"):
            content = content_bytes.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Split content into chunks
        chunks = _split_text(content)

        # Prepare metadata
        metadata = []
        base_metadata = {"filename": file.filename, "content_type": file.content_type}
        for i, chunk in enumerate(chunks):
            chunk_metadata = {**base_metadata, "chunk_id": i}
            metadata.append(chunk_metadata)

        # Add to RAG system
        service.add_documents(chunks, metadata)

        return RAGUploadResponse(
            success=True,
            message=f"Successfully uploaded {file.filename} with {len(chunks)} chunks",
        )

    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGQueryResponse)
async def query_documents(
    request: RAGQueryRequest, service: ChatService = Depends(get_chat_service)
) -> RAGQueryResponse:
    """Query RAG system for relevant documents."""
    try:
        if not service.is_rag_available():
            raise HTTPException(status_code=400, detail="RAG not enabled")

        results = service.retrieve_documents(request.query, request.top_k or 5)

        return RAGQueryResponse(results=results, query=request.query)

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def rag_stats(service: ChatService = Depends(get_chat_service)) -> Dict[str, Any]:
    """Get RAG system statistics."""
    try:
        stats = service.get_rag_stats()
        return stats

    except Exception as e:
        logger.error(f"RAG stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _split_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Simple text splitting into chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at sentence boundary
        chunk = text[start:end]
        last_sentence = max(chunk.rfind("."), chunk.rfind("!"), chunk.rfind("?"))

        if last_sentence > chunk_size // 2:  # Only break if sentence is not too short
            end = start + last_sentence + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [chunk for chunk in chunks if chunk.strip()]
