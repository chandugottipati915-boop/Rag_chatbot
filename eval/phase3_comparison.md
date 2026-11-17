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

## Part C: LangGraph pipeline + multi-turn (2026-11-17)

Graph: rewrite → retrieve (hybrid + rerank) → grade → reformulate once → generate / abstain.
New golden set: `goldens/multiturn_golden.jsonl` (14 conversations), script `eval/eval_multiturn.py`.

### Multi-turn, first run (rewrite with paper filter)

| Run | Paper hit | Recall | Refusal ok | Rewrite | Faithful | Behavior |
|---|---|---|---|---|---|---|
| with history | 0.85 | 0.81 | 0.71 | 0.85 | 0.83 | 0.62 |
| no history | 0.62 | 0.54 | 0.93 | — | 0.85 | 0.55 |

Findings: the paper filter caused both wrong-paper failures (m08: RAGAS filtered to CRAG;
m12: "it" resolved to the older subject, Self-RAG) and was never needed for a success
(m03/m04/m05/m10 found the right paper without it). m13 (false premise in a follow-up) refused.

Fixes: paper filter off by default (`REWRITE_PAPER_FILTER`), rewrite rules for "most recent
subject" / "they" / "the two", a premise check before refusing in `system.txt`, and `filter ok`
now only counts a wrong filter as a failure.

### Multi-turn, after the fixes

| Run | Paper hit | Recall | Refusal ok | Rewrite | Faithful | Behavior |
|---|---|---|---|---|---|---|
| **with history** | **1.00** | **0.96** | **1.00** | **0.90** | **0.88** | **0.82** |
| no history | 0.62 | 0.54 | 0.93 | — | 0.94 | 0.54 |

Rewriting adds +0.38 paper hit and +0.28 correct behavior on follow-ups.
Remaining: comparisons sometimes retrieve one side only (m06, behavior 0.48).

### Generator eval after the fixes (direct pipeline: retrieve → generate)

| Group | Faithfulness | Relevancy | Correct behavior |
|---|---|---|---|
| answer (33) | 0.902 | 0.966 | 0.715 (22/33) |
| correct_premise (4) | 0.938 | 0.253 | 0.372 (1/4) |
| abstain (4) | 0.750 | 0.000 | 1.000 (4/4) |
| clarify_or_cover (2) | 0.700 | 0.944 | 0.184 (0/2) |

No regressions: unanswerable questions still refused 4/4; false refusals 1 (g33).
The premise check fixed m13 (multi-turn) but not single-question false premises (still 1/4).

### Generator eval through the full graph (2026-11-17)

| Group | Faithfulness | Relevancy | Correct behavior |
|---|---|---|---|
| answer (33) | **0.928** | 0.948 | 0.703 (22/33) |
| correct_premise (4) | 0.917 | 0.202 | 0.401 (1/4) |
| abstain (4) | 1.000 | 0.000 | 1.000 (4/4) |
| clarify_or_cover (2) | 1.000 | 1.000 | 0.191 (0/2) |

vs. the direct pipeline: faithfulness +0.026 (grading keeps irrelevant chunks out), relevancy and
correct behavior within noise. False refusals 1 (g33), unanswerable not refused 0.
Retries used: g34 (false premise) and g40 (off-topic).

### Latency, full graph (laptop)

| Questions | p50 | p95 | max | Notes |
|---|---|---|---|---|
| Paper questions (33) | 4441 ms | 5652 ms | 8793 ms | retries used 2/33 |
| Off-topic (5) | 5466 ms | 5965 ms | 5965 ms | refused 4/5 |

The off-topic set had one question where some chunk passed grading, so it was answered or
refused by the generator instead of the abstain step; worth checking which one.

### Phase 3 summary

| | Start of Phase 3 | End of Phase 3 |
|---|---|---|
| Retrieval precision / recall | 0.708 / 0.956 | **0.881 / 0.969** |
| Generator faithfulness / relevancy (answerable) | 0.899 / 0.843 | **0.928 / 0.948** |
| False refusals | 5 | **1** |
| Unanswerable refused | 4/4 | 4/4 |
| Follow-up paper hit (multi-turn) | not supported | **1.00** (0.62 without history) |
| Latency (full answer, p50) | not measured (retrieval alone: 1.3 s) | 4.4 s |

Open issues for later: single-question false premises (1/4), ambiguous questions (0/2),
one-sided comparisons (m06), the "21 million documents" detail (g33 / m05), off-topic latency.
