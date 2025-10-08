"""FastAPI application entry point."""

# backend/app/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .settings import settings
from .utils.logging import setup_logging
from .routers import chat_router, rag_router, health_router

# Setup logging
setup_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    log_file=settings.BASE_DIR / "logs" / "app.log" if not settings.DEBUG else None,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting LLM Switch Chat API")
    logger.info(f"Backend: {settings.MODEL_BACKEND}")
    logger.info(f"Model: {settings.MODEL_ID}")
    logger.info(f"RAG enabled: {settings.ENABLE_RAG}")

    yield

    logger.info("Shutting down LLM Switch Chat API")


# Create FastAPI app
app = FastAPI(
    title="LLM Switch Chat",
    description="Multi-backend LLM chat API with RAG support",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(rag_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Switch Chat API",
        "backend": settings.MODEL_BACKEND,
        "model": settings.MODEL_ID,
        "rag_enabled": settings.ENABLE_RAG,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )
