import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Latency for the current settings (RETRIEVAL_MODE, RERANK, EMBEDDING_PROVIDER, MAX_RETRIES).

  uv run python scripts/bench_latency.py            # retrieval only (embedding + search + rerank)
  uv run python scripts/bench_latency.py --graph    # full graph, as /chat runs it (uses the LLM: costs a few cents)
"""

import json
import statistics
import time

from rag.config import PROJECT_ROOT, settings
from rag.retrieval import retrieve

OFF_TOPIC = [
    "How many minutes are in an hour?",
    "What is the capital of France?",
    "Who won the most recent football World Cup?",
    "What is a good recipe for pancakes?",
    "How do I reset my router?",
]


def stats(times: list[float]) -> str:
    times = sorted(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    return f"p50={statistics.median(times):>5.0f} ms  p95={p95:>5.0f} ms  max={times[-1]:>5.0f} ms  (n={len(times)})"


def bench(fn, questions: list[str]) -> tuple[list[float], list]:
    times, outputs = [], []
    for q in questions:
        t0 = time.perf_counter()
        outputs.append(fn(q))
        times.append((time.perf_counter() - t0) * 1000)
    return times, outputs


questions = [json.loads(l)["question"]
             for l in (PROJECT_ROOT / "goldens" / "retrieval_golden.jsonl").read_text().splitlines() if l.strip()]
print(f"embeddings={settings.embedding_provider}  mode={settings.retrieval_mode}  rerank={settings.rerank}"
      f"  max_retries={settings.max_retries}")

if "--graph" not in sys.argv:
    retrieve(questions[0])                      # warm-up: loads models, opens connections
    times, _ = bench(retrieve, questions)
    print(f"retrieval latency   {stats(times)}")
    sys.exit()

from rag.graph.build import get_graph

graph = get_graph()
graph.invoke({"question": questions[0]})     # warm-up
run = lambda q: graph.invoke({"question": q})

times, states = bench(run, questions)
retried = sum(1 for s in states if s.get("retries"))
print(f"graph, paper questions   {stats(times)}   retries used: {retried}/{len(states)}")

times, states = bench(run, OFF_TOPIC)
refused = sum(1 for s in states if not s.get("chunks"))
print(f"graph, off-topic         {stats(times)}   refused: {refused}/{len(states)}")
