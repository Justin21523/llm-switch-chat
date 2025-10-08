"""Health check endpoints."""

# backend/app/routers/health.py
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {"status": "healthy", "message": "LLM Switch Chat API is running"}


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check."""
    # TODO: Add actual backend availability checks
    return {"status": "ready", "message": "API is ready to serve requests"}
