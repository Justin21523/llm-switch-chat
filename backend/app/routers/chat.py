"""Chat API endpoints."""

# backend/app/routers/chat.py
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Iterator
import json

from ..schemas.chat import ChatRequest, ChatResponse, StreamChunk
from ..core.chat_service import ChatService
from ..deps import get_chat_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest, service: ChatService = Depends(get_chat_service)
) -> ChatResponse:
    """Generate chat response."""
    try:
        response = service.chat(request)
        return response

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(
    request: ChatRequest, service: ChatService = Depends(get_chat_service)
):
    """Generate streaming chat response."""

    def generate_stream() -> Iterator[str]:
        try:
            for chunk in service.chat_stream(request):
                # Format as Server-Sent Events
                chunk_data = chunk.dict()
                yield f"data: {json.dumps(chunk_data)}\n\n"

                if chunk.finished:
                    break

        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            error_chunk = StreamChunk(token="", finished=True)
            yield f"data: {json.dumps(error_chunk.dict())}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/status")
async def chat_status(
    service: ChatService = Depends(get_chat_service),
) -> Dict[str, Any]:
    """Get chat service status."""
    try:
        backend_available = service.backend.is_available()
        rag_stats = service.get_rag_stats()

        return {"backend_available": backend_available, "rag": rag_stats}

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
