"""RAG graph.

    START → rewrite → retrieve → grade ──┬── relevant chunks           → generate → END
                                ▲        ├── none relevant, retry left → reformulate ┐
                                └────────┼───────────────────────────────────────────┘
                                         └── none relevant, no retry   → abstain  → END

Two compiled versions:
- get_graph():           the full pipeline, used by /chat, scripts and evals.
- get_retrieval_graph(): stops where generation would start, used by /chat/stream so the
                         API can send the graded sources first and then stream the tokens.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from rag.graph import nodes
from rag.graph.state import RAGState


def _build(with_generate: bool):
    builder = StateGraph(RAGState)

    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("reformulate", nodes.reformulate)
    builder.add_node("abstain", nodes.abstain)
    if with_generate:
        builder.add_node("generate", nodes.generate)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    # decision point: route_after_grade returns "generate", "reformulate" or "abstain".
    # Without a generate node, "generate" means "stop here, the caller will generate".
    builder.add_conditional_edges("grade", nodes.route_after_grade, {
        "generate": "generate" if with_generate else END,
        "reformulate": "reformulate",
        "abstain": "abstain",
    })
    builder.add_edge("reformulate", "retrieve")   # the loop: search again with the new query
    builder.add_edge("abstain", END)
    if with_generate:
        builder.add_edge("generate", END)

    return builder.compile()


@lru_cache
def get_graph():
    """Full pipeline: rewrite → retrieve → grade → (reformulate) → generate / abstain."""
    return _build(with_generate=True)


@lru_cache
def get_retrieval_graph():
    """Everything up to generation. If it abstained, the state already has an answer."""
    return _build(with_generate=False)
