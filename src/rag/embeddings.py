from functools import lru_cache

from rag.config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=settings.embed_batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE models retrieve better when queries carry this instruction
        return self.model.encode(
            BGE_QUERY_PREFIX + text, normalize_embeddings=True
        ).tolist()


class OpenAIEmbedder:
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model_name
        self.dim = self.DIMS[model_name]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), settings.embed_batch_size):
            batch = texts[i : i + settings.embed_batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder():
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model)
    return LocalEmbedder(settings.local_embedding_model)