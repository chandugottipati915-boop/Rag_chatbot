"""C1 graph: rewrite → retrieve → generate.

C2 adds a grade node with a conditional edge (generate vs. abstain),
C3 adds one retry with a reformulated query.
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
    builder.add_node("generate", nodes.generate)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
