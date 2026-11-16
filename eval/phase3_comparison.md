# Phase 3: retrieval setups compared

All runs: same golden sets, top_k = 5, gpt-4o-mini generator and judge, new system prompt,
survey section fix applied (re-ingested). Logs: `eval/results/hybrid_run_20261112-113433.log`,
`eval/results/eval_openai_20261112-114159.log`.

## Retrieval (30 answerable questions)

| Setup | Contextual Precision | Contextual Recall |
|---|---|---|
| Baseline: BGE dense (before survey fix) | 0.708 (19/30) | 0.956 (28/30) |
| BGE hybrid | 0.830 (23/30) | 0.936 (27/30) |
| OpenAI dense | 0.830 (23/30) | 0.903 (26/30) |
| **OpenAI hybrid** | **0.838 (24/30)** | **0.936 (28/30)** |

## Generator (should-answer group, n = 33)

| Setup | Faithfulness | Answer Relevancy | Correct Behavior | False refusals |
|---|---|---|---|---|
| BGE dense (after prompt fix) | 0.933 | 0.951 | 0.687 | 0 |
| BGE hybrid | 0.906 | 0.937 | 0.691 | 1 (g33) |
| OpenAI dense | 0.926 | 0.939 | 0.697 | 1 (g33) |
| OpenAI hybrid | 0.904 | 0.935 | 0.701 | 1 (g33) |

## Generator (behavior groups, Correct Behavior score)

| Setup | False premise (4) | Unanswerable (4) | Ambiguous (2) |
|---|---|---|---|
| BGE dense (after prompt fix) | 0.305 (1/4) | 0.967 (4/4) | 0.174 (0/2) |
| BGE hybrid | 0.308 (1/4) | 0.750 (3/4) | 0.190 (0/2) |
| OpenAI dense | 0.360 (1/4) | 1.000 (4/4) | 0.195 (0/2) |
| OpenAI hybrid | 0.326 (1/4) | 1.000 (4/4) | 0.151 (0/2) |

## Notes

- The jump from 0.71 to ~0.83 precision happened in every setup, so it most likely comes from the
  survey section fix + re-ingest, not from hybrid search or OpenAI embeddings. Confirm with a
  BGE dense retrieval re-run: `EMBEDDING_PROVIDER=local bash scripts/run_eval.sh dense`.
- Differences between the three new setups are small (≤ 0.01 precision, ≤ 0.03 recall) and within
  the judge's run-to-run noise. Hybrid recovers the recall that OpenAI dense loses (0.903 → 0.936).
- g33 (RAG index size, "21 million documents"): answered by BGE dense, refused by the other three →
  retrieval ranking regression for that chunk.
- g40 (Turing Award) with BGE hybrid: answered with an unrelated Nobel Prize fact from a source
  instead of refusing.
- g12 (Self-RAG base models) with OpenAI hybrid: answered "T5-large", which is CRAG's evaluator,
  not Self-RAG's generator (Llama 2 7B/13B) → cross-paper mix-up. Still a retrieval miss in all setups.
- False premises (1/4) and ambiguous questions (0/2) are unchanged by retrieval: these are
  generator/prompt problems, not retrieval problems.

## Run: 2026-11-12 12:01 — OpenAI hybrid + rerank (bge-reranker-base, 20 candidates → top 5)

| Setup | Precision | Recall | Faithfulness* | Answer Rel.* | Correct Behavior* | Retrieval p50 / p95 |
|---|---|---|---|---|---|---|
| Baseline: BGE dense | 0.708 | 0.956 | 0.899 | 0.843 | — | not measured |
| OpenAI hybrid | 0.838 | 0.936 | 0.904 | 0.935 | 0.701 | not measured |
| **OpenAI hybrid + rerank** | **0.881 (26/30)** | **0.969 (29/30)** | 0.908 | **0.960** | 0.701 | 1333 / 1605 ms |

\* generator scores on the should-answer group (n = 33). Log: `eval/results/eval_openai_20261112-120124.log`.

Reranking gives the best retrieval of every setup: precision +0.04 and recall +0.03 over hybrid,
and both targets are met (precision ≥ 0.85, recall ≥ 0.95). Cost: ~1.3 s per query for
retrieval (OpenAI query embedding + cross-encoder over 20 chunks on the laptop).
