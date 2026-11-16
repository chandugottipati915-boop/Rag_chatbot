import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Run the LangGraph pipeline on a single question and on a follow-up.

  uv run python scripts/try_graph.py
  uv run python scripts/try_graph.py --diagram     # print the graph as Mermaid
"""

import logging

from rag.graph.build import get_graph

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
graph = get_graph()

if "--diagram" in sys.argv:
    print(graph.get_graph().draw_mermaid())   # paste into https://mermaid.live
    sys.exit()


def show(result: dict) -> None:
    print(f"\nQ:      {result['question']}")
    print(f"query:  {result['query']}   (paper filter: {result.get('paper_filter')})")
    print(f"graded: kept {len(result.get('chunks', []))}/{len(result.get('retrieved', []))} chunks")
    print(f"answer: {result['answer']}")
    for i, c in enumerate(result.get("chunks", []), 1):
        print(f"  [{i}] {c.citation}")


# 1) a standalone question: no history, so no rewrite call
first = graph.invoke({"question": "What is Self-RAG?"})
show(first)

# 2) a follow-up: "it" must be resolved to Self-RAG before retrieval
show(graph.invoke({
    "question": "How is it trained?",
    "history": [
        {"role": "user", "content": first["question"]},
        {"role": "assistant", "content": first["answer"]},
    ],
}))

# 3) C2 checks: grading drops wrong-paper chunks, refuses off-topic, keeps premise-correcting chunks
for q in [
    "Which base models does Self-RAG fine-tune?",                    # g12: CRAG chunks should be dropped
    "How many minutes are in an hour?",                              # off-topic: 0 kept → abstain
    "Why did the original RAG paper use GPT-2 as its generator?",    # false premise: BART chunk kept
]:
    show(graph.invoke({"question": q}))

