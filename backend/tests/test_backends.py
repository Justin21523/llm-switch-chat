"""Test backend implementations."""

# backend/tests/test_backends.py
import pytest
from unittest.mock import Mock, patch
from app.backends.registry import registry


class TestBackendRegistry:
    """Test backend registry."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = registry.list_backends()

        assert "hf" in backends
        assert "llamacpp" in backends
        assert "openai" in backends

    def test_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend type"):
            registry.get_backend("unknown_backend")


class TestTransformersBackend:
    """Test Transformers backend."""

    @patch("app.backends.transformers_backend.AutoTokenizer")
    @patch("app.backends.transformers_backend.AutoModelForCausalLM")
    def test_initialization(self, mock_model_class, mock_tokenizer_class):
        """Test backend initialization."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create backend
        backend = registry.get_backend("hf", model_id="test-model", device="cpu")

        assert backend.is_available()
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()


class TestLlamaCppBackend:
    """Test llama.cpp backend."""

    @patch("app.backends.llamacpp_backend.Llama")
    def test_initialization(self, mock_llama_class):
        """Test backend initialization."""
        mock_llama = Mock()
        mock_llama_class.return_value = mock_llama

        backend = registry.get_backend("llamacpp", gguf_path="test.gguf", n_ctx=2048)

        assert backend.is_available()
        mock_llama_class.assert_called_once_with(
            model_path="test.gguf", n_ctx=2048, n_gpu_layers=-1, verbose=False
        )


class TestOpenAIBackend:
    """Test OpenAI backend."""

    @patch("app.backends.openai_backend.OpenAI")
    def test_initialization(self, mock_openai_class):
        """Test backend initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        backend = registry.get_backend(
            "openai",
            model_id="gpt-3.5-turbo",
            base_url="http://localhost:8001",
            api_key="test-key",
        )

        assert backend.is_available()
        mock_openai_class.assert_called_once_with(
            base_url="http://localhost:8001", api_key="test-key"
        )


if __name__ == "__main__":
    pytest.main([__file__])
