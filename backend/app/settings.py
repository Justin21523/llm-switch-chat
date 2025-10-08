"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field
import yaml


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Model Backend Settings
    MODEL_BACKEND: str = Field(
        default="hf", env="MODEL_BACKEND"
    )  # hf, llamacpp, openai
    MODEL_ID: str = Field(default="Qwen/Qwen2-7B-Instruct", env="MODEL_ID")

    # HuggingFace Settings
    HF_DEVICE: str = Field(default="auto", env="HF_DEVICE")
    HF_TORCH_DTYPE: str = Field(default="auto", env="HF_TORCH_DTYPE")

    # llama.cpp Settings
    GGUF_PATH: Optional[str] = Field(default=None, env="GGUF_PATH")
    LLAMACPP_N_CTX: int = Field(default=4096, env="LLAMACPP_N_CTX")
    LLAMACPP_N_GPU_LAYERS: int = Field(default=-1, env="LLAMACPP_N_GPU_LAYERS")

    # OpenAI-compatible Settings
    OPENAI_BASE_URL: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    OPENAI_API_KEY: Optional[str] = Field(default="sk-default", env="OPENAI_API_KEY")

    # Generation Settings
    MAX_TOKENS: int = Field(default=1024, env="MAX_TOKENS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    TOP_P: float = Field(default=0.9, env="TOP_P")

    # RAG Settings
    ENABLE_RAG: bool = Field(default=False, env="ENABLE_RAG")
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )
    VECTOR_STORE_PATH: str = Field(
        default="./data/vectorstore", env="VECTOR_STORE_PATH"
    )
    RAG_TOP_K: int = Field(default=5, env="RAG_TOP_K")

    # Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    MODELS_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "models"
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load YAML config if exists
        yaml_path = self.BASE_DIR / "configs" / "app.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
                for key, value in yaml_config.items():
                    if (
                        not hasattr(self, key)
                        or getattr(self, key) == self.__fields__[key].default
                    ):
                        setattr(self, key, value)

        # Ensure directories exist
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.VECTOR_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
