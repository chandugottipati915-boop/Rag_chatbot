import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector,
    MatchValue, PointStruct, VectorParams,
)

from rag.config import settings

client = QdrantClient(url=settings.qdrant_url, timeout=5)  # fail fast when Qdrant is down
COLL = settings.collection_name


def ensure_collection(dim: int) -> None:
    if client.collection_exists(COLL):
        return
    client.create_collection(
        collection_name=COLL,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    # indexes make filtering by these fields fast
    for field in ("arxiv_id", "doc_hash", "section"):
        client.create_payload_index(COLL, field_name=field, field_schema="keyword")


def _match(key: str, value: str) -> Filter:
    return Filter(must=[FieldCondition(key=key, match=MatchValue(value=value))])


def doc_is_indexed(doc_hash: str) -> bool:
    return client.count(COLL, count_filter=_match("doc_hash", doc_hash)).count > 0


def delete_doc(arxiv_id: str) -> None:
    client.delete(COLL, points_selector=FilterSelector(filter=_match("arxiv_id", arxiv_id)))


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            # Qdrant IDs must be ints or UUIDs → deterministic UUID from our chunk id
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
            vector=v,
            payload={"text": c["text"], **c["metadata"]},
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(COLL, points=points)


def search(query_vector: list[float], top_k: int = settings.top_k, arxiv_id: str | None = None):
    return client.query_points(
        COLL,
        query=query_vector,
        limit=top_k,
        query_filter=_match("arxiv_id", arxiv_id) if arxiv_id else None,
        with_payload=True,
    ).points