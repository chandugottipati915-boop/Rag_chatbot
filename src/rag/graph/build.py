"""RAG graph.

    START → rewrite → retrieve → grade ──┬── relevant chunks           → generate → END
                                ▲        ├── none relevant, retry left → reformulate ┐
                                └────────┼───────────────────────────────────────────┘
                                         └── none relevant, no retry   → abstain  → END
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from rag.graph import nodes
from rag.graph.state import RAGState


@lru_cache
def get_graph():
    builder = StateGraph(RAGState)

    builder.add_node("rewrite", nodes.rewrite)
    builder.add_node("retrieve", nodes.retrieve)
    builder.add_node("grade", nodes.grade)
    builder.add_node("generate", nodes.generate)
    builder.add_node("abstain", nodes.abstain)
    builder.add_node("reformulate", nodes.reformulate)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    # decision point: route_after_grade returns "generate", "reformulate" or "abstain"
    builder.add_conditional_edges("grade", nodes.route_after_grade, ["generate", "reformulate", "abstain"])
    builder.add_edge("reformulate", "retrieve")   # the loop: search again with the new query
    builder.add_edge("generate", END)
    builder.add_edge("abstain", END)

    return builder.compile()
