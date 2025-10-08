"""llama.cpp backend using llama-cpp-python."""

### backend/app/backends/llamacpp_backend.py
import logging
from typing import Iterator, Optional, Dict, Any, List
from ..backends.base import BaseBackend
from ..schemas.chat import ChatMessage

logger = logging.getLogger(__name__)


class LlamaCppBackend(BaseBackend):
    """llama.cpp backend."""

    def __init__(
        self, gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, **kwargs
    ):
        super().__init__(**kwargs)
        self.gguf_path = gguf_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llama = None
        self._initialize()

    def _initialize(self):
        """Initialize llama.cpp model."""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model: {self.gguf_path}")

            self._llama = Llama(
                model_path=self.gguf_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )

            logger.info("GGUF model loaded successfully")

        except ImportError:
            logger.error(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate response from prompt."""
        if not self.is_available():
            raise RuntimeError("Model not available")

        try:
            generation_kwargs = {
                "max_tokens": max_tokens or 1024,
                "temperature": temperature or 0.7,
                "top_p": top_p or 0.9,
                "stop": kwargs.get("stop", []),
                "echo": False,
            }

            response = self._llama(prompt, **generation_kwargs)

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Iterator[str]:
        """Generate streaming response from prompt."""
        if not self.is_available():
            raise RuntimeError("Model not available")

        try:
            generation_kwargs = {
                "max_tokens": max_tokens or 1024,
                "temperature": temperature or 0.7,
                "top_p": top_p or 0.9,
                "stop": kwargs.get("stop", []),
                "stream": True,
            }

            for chunk in self._llama(prompt, **generation_kwargs):
                if chunk["choices"][0]["text"]:
                    yield chunk["choices"][0]["text"]

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if backend is available."""
        return self._llama is not None
