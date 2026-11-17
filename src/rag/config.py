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
    retrieval_mode: Literal["dense", "hybrid"] = "dense"   # hybrid = dense + BM25, fused with RRF
    prefetch_k: int = 20                                   # candidates per retriever before fusion
    rerank: bool = False                                   # re-order candidates with a cross-encoder
    rerank_candidates: int = 20                            # how many candidates the reranker sees
    reranker_model: str = "BAAI/bge-reranker-base"

    # llm
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # graph
    max_retries: int = 1                                   # reformulate-and-search attempts before abstaining
    # Let the rewrite restrict retrieval to one paper. Off by default: in the multi-turn eval
    # a wrong guess hid the answer (m08, m12) and hybrid + rerank found the right paper anyway.
    rewrite_paper_filter: bool = False

    @property
    def collection_name(self) -> str:
        # separate collection per provider (vector sizes differ) and per mode
        # (hybrid points carry an extra sparse vector), so you can compare them side by side
        base = f"papers_{self.embedding_provider}"
        return f"{base}_hybrid" if self.retrieval_mode == "hybrid" else base


settings = Settings()