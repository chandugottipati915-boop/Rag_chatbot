from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# project root = the folder containing pyproject.toml (src/rag/config.py -> ../../)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # absolute path: works no matter which folder you run a script from
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # paths
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"

    # embeddings
    embedding_provider: Literal["local", "openai"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    embed_batch_size: int = 64

    # qdrant
    qdrant_url: str = "http://localhost:6333"

    # retrieval
    top_k: int = 5

    # llm
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    @property
    def collection_name(self) -> str:
        # separate collection per provider: vector sizes differ
        return f"papers_{self.embedding_provider}"


settings = Settings()