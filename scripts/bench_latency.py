import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Retrieval latency for the current settings (RETRIEVAL_MODE, RERANK, EMBEDDING_PROVIDER).

  uv run python scripts/bench_latency.py
  RETRIEVAL_MODE=hybrid RERANK=true uv run python scripts/bench_latency.py
"""

import json
import statistics
import time

from rag.config import PROJECT_ROOT, settings
from rag.retrieval import retrieve

questions = [json.loads(l)["question"]
             for l in (PROJECT_ROOT / "goldens" / "retrieval_golden.jsonl").read_text().splitlines() if l.strip()]

retrieve(questions[0])                      # warm-up: loads models, opens connections
times = []
for q in questions:
    t0 = time.perf_counter()
    retrieve(q)
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
print(f"embeddings={settings.embedding_provider}  mode={settings.retrieval_mode}  rerank={settings.rerank}"
      f"  n={len(times)}")
print(f"retrieval latency  p50={statistics.median(times):.0f} ms  p95={p95:.0f} ms  max={times[-1]:.0f} ms")
