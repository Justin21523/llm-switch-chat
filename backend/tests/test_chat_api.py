"""Test chat API endpoints."""

# backend/tests/test_chat_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.schemas.chat import ChatMessage

client = TestClient(app)


class TestChatAPI:
    """Test chat API endpoints."""

    @patch("app.deps.get_chat_service")
    def test_chat_endpoint(self, mock_get_service):
        """Test basic chat endpoint."""
        # Setup mock service
        mock_service = Mock()
        mock_service.chat.return_value.response = "Hello, how can I help you?"
        mock_service.chat.return_value.rag_context = None
        mock_get_service.return_value = mock_service

        # Make request
        response = client.post(
            "/chat/", json={"message": "Hello", "history": [], "use_rag": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["response"] == "Hello, how can I help you?"

    @patch("app.deps.get_chat_service")
    def test_chat_with_history(self, mock_get_service):
        """Test chat with conversation history."""
        mock_service = Mock()
        mock_service.chat.return_value.response = (
            "I remember our previous conversation."
        )
        mock_service.chat.return_value.rag_context = None
        mock_get_service.return_value = mock_service

        history = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]

        response = client.post(
            "/chat/", json={"message": "What's my name?", "history": history}
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data

    @patch("app.deps.get_chat_service")
    def test_chat_status(self, mock_get_service):
        """Test chat status endpoint."""
        mock_service = Mock()
        mock_service.backend.is_available.return_value = True
        mock_service.get_rag_stats.return_value = {
            "available": True,
            "document_count": 5,
        }
        mock_get_service.return_value = mock_service

        response = client.get("/chat/status")

        assert response.status_code == 200
        data = response.json()
        assert data["backend_available"] is True
        assert data["rag"]["available"] is True


class TestRAGAPI:
    """Test RAG API endpoints."""

    @patch("app.deps.get_chat_service")
    def test_rag_upload(self, mock_get_service):
        """Test RAG document upload."""
        mock_service = Mock()
        mock_service.is_rag_available.return_value = True
        mock_service.add_documents = Mock()
        mock_get_service.return_value = mock_service

        response = client.post(
            "/rag/upload",
            json={
                "content": "This is a test document.",
                "metadata": {"source": "test"},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_service.add_documents.assert_called_once()

    @patch("app.deps.get_chat_service")
    def test_rag_query(self, mock_get_service):
        """Test RAG document query."""
        mock_service = Mock()
        mock_service.is_rag_available.return_value = True
        mock_service.retrieve_documents.return_value = [
            {"text": "Relevant document", "score": 0.8}
        ]
        mock_get_service.return_value = mock_service

        response = client.post("/rag/query", json={"query": "test query", "top_k": 5})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1


class TestHealthAPI:
    """Test health check endpoints."""

    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/health/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self):
        """Test readiness check."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


if __name__ == "__main__":
    pytest.main([__file__])
