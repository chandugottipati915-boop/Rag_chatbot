# Generator Eval Notes

## Run: 2026-11-11 13:35 — baseline generator (retrieve → generate)

- Script: `eval/eval_generator.py` (DeepEval)
- Judge: `gpt-4o-mini` · `top_k = 5` · threshold = 0.7
- Metrics: Faithfulness (claims vs. retrieval context), Answer Relevancy (answer vs. question)
- Golden set: `goldens/generator_golden.jsonl`, 43 questions (33 answerable, 4 false-premise, 4 unanswerable, 2 ambiguous)
- Generated answers: `eval/results/generator_answers_20261111-133513.jsonl`
- Scores: `.deepeval/.latest_test_run.json`
- Judge run time ~23.9 s · judge cost ~$0.047

### Summary

| Metric | Mean | Passed (≥ 0.7) |
|---|---|---|
| Faithfulness | **0.853** | 38 / 43 |
| Answer Relevancy | **0.694** | 30 / 43 |

Test cases passing both metrics: 30 / 43 (13 failed).

| Expected behavior | n | Faithfulness | Answer Relevancy |
|---|---|---|---|
| answer | 33 | 0.899 (31 pass) | 0.843 (28 pass) |
| correct_premise | 4 | 0.500 (2 pass) | 0.000 (0 pass) |
| abstain | 4 | 0.750 (3 pass) | 0.000 (0 pass) |
| clarify_or_cover | 2 | 1.000 (2 pass) | 1.000 (2 pass) |

| Category | n | Faithfulness | Answer Relevancy |
|---|---|---|---|
| factual | 12 | 0.889 | 0.833 |
| explanation | 8 | 0.979 | 0.979 |
| comparison | 5 | 0.933 | 0.800 |
| format | 4 | 0.688 | 0.750 |
| numeric | 4 | 0.938 | 0.750 |
| false_premise | 4 | 0.500 | 0.000 |
| unanswerable | 4 | 0.750 | 0.000 |
| ambiguous | 2 | 1.000 | 1.000 |

### Failures (all 13 are the answer "I don't know based on the provided papers.")

| ID | Category | Faith. | Relev. | Question | Verdict |
|---|---|---|---|---|---|
| g09 | factual | 1.00 | 0.00 | What does GraphRAG pre-compute before any question is asked? | **False abstention** |
| g12 | factual | 0.00 | 0.00 | Which base models does Self-RAG fine-tune? | **False abstention** |
| g22 | comparison | 1.00 | 0.00 | Compare DPR and HyDE in how they get a good query representation. | **False abstention** |
| g27 | format | 0.00 | 0.00 | List the critique tokens Self-RAG uses to judge passages and its own output. | **False abstention** |
| g33 | numeric | 1.00 | 0.00 | Roughly how many documents are in RAG's retrieval index? | **False abstention** |
| g34 | false_premise | 0.00 | 0.00 | Why did the original RAG paper use GPT-2 as its generator? | Should correct premise (BART) |
| g35 | false_premise | 1.00 | 0.00 | How does ColBERT compress each passage into a single vector? | Should correct premise (multi-vector) |
| g36 | false_premise | 0.00 | 0.00 | Which RL algorithm does Self-RAG use to learn its reflection tokens? | Should correct premise (no RL) |
| g37 | false_premise | 1.00 | 0.00 | Since BERT is a decoder-only model, how does it generate long text? | Should correct premise (encoder-only) |
| g38 | unanswerable | 0.00 | 0.00 | Which vector database is best for production RAG systems today? | Correct abstain, metric penalizes |
| g39 | unanswerable | 1.00 | 0.00 | How many dollars did it cost to train GPT-3? | Correct abstain, metric penalizes |
| g40 | unanswerable | 1.00 | 0.00 | Who received the most recent Turing Award? | Correct abstain, metric penalizes |
| g41 | unanswerable | 1.00 | 0.00 | What chunk overlap does LangChain recommend for RAG? | Correct abstain, metric penalizes |

Below-1.0 but passing: g08, g10, g21, g25 (Faith. 0.83), g13 (0.83 / 0.83), g28, g32 (Faith. 0.75).

### Observations

- **When it answers, it's good:** on the 28 answered questions, both metrics are almost always ≥ 0.83; explanation questions score ~0.98.
- **Every failure is an abstention.** The judge scores "I don't know" as 0 relevancy, and sometimes 0 faithfulness (g12, g27, g34, g36, g38: an artifact of the judge, not a hallucination).
- **5 false abstentions on answerable questions (g09, g12, g22, g27, g33):** likely retrieval misses (compare with the retrieval eval) or an over-strict refusal prompt. This is the main real problem.
- **False-premise questions (g34–g37):** the model refuses instead of correcting the premise. Desired behavior is "Actually, RAG uses BART…". Needs a prompt change.
- **Unanswerable questions (g38–g41):** abstaining is the *right* behavior, so these 4 "failures" are metric noise. Answer Relevancy / Faithfulness don't fit `abstain` / `correct_premise` cases.
- Next ideas:
  - Check retrieved chunks for g09/g12/g22/g27/g33 to separate retrieval misses from over-refusal.
  - Update the generation prompt to correct false premises instead of refusing.
  - Score `abstain` / `correct_premise` cases with a custom GEval metric (e.g. "did it correctly abstain / correct the premise?") instead of relevancy/faithfulness, or report them separately.
