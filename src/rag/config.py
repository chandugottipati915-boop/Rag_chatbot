from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # paths
    raw_dir: Path = Path("data/raw")

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

    @property
    def collection_name(self) -> str:
        # separate collection per provider: vector sizes differ
        return f"papers_{self.embedding_provider}"


settings = Settings()