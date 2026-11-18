import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector,
    MatchValue, PointStruct, VectorParams,
)

from rag.config import settings

# fail fast when Qdrant is down; the API key is only set for Qdrant Cloud
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)
COLL = settings.collection_name
HYBRID = settings.retrieval_mode == "hybrid"

# Hybrid collections use named vectors: a dense embedding plus a sparse BM25 vector.
# Qdrant computes BM25 itself from the raw text (needs Qdrant >= 1.15.3).
DENSE, SPARSE = "dense", "bm25"
BM25_MODEL = "Qdrant/bm25"


def ensure_collection(dim: int) -> None:
    if client.collection_exists(COLL):
        return
    if HYBRID:
        client.create_collection(
            collection_name=COLL,
            vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            # IDF: rare terms ("CRAG", "MaxSim") weigh more than common ones ("the", "model")
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    else:
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


def _vector(text: str, dense: list[float]):
    if HYBRID:
        return {DENSE: dense, SPARSE: models.Document(text=text, model=BM25_MODEL)}
    return dense


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(
            # Qdrant IDs must be ints or UUIDs → deterministic UUID from our chunk id
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["id"])),
            vector=_vector(c["text"], v),
            payload={"text": c["text"], **c["metadata"]},
        )
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(COLL, points=points)


def search(
    query_vector: list[float],
    top_k: int = settings.top_k,
    arxiv_id: str | None = None,
    query_text: str | None = None,
):
    flt = _match("arxiv_id", arxiv_id) if arxiv_id else None

    if not HYBRID:
        return client.query_points(
            COLL, query=query_vector, limit=top_k, query_filter=flt, with_payload=True,
        ).points

    if not query_text:
        raise ValueError("hybrid search needs the query text for BM25")
    # Run dense and BM25 searches, then merge their rankings with Reciprocal Rank Fusion.
    # Note: RRF scores are rank-based (not cosine similarities), so they aren't comparable
    # with dense scores and min_score thresholds from dense mode don't apply.
    return client.query_points(
        COLL,
        prefetch=[
            models.Prefetch(query=query_vector, using=DENSE, limit=settings.prefetch_k, filter=flt),
            models.Prefetch(query=models.Document(text=query_text, model=BM25_MODEL),
                            using=SPARSE, limit=settings.prefetch_k, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
