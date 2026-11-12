# Retrieval Eval Notes

## Run: 2026-11-10 22:18 — baseline retriever

- Script: `eval/eval_retriever.py` (DeepEval)
- Judge: `gpt-4o-mini` · `top_k = 5` · threshold = 0.7
- Golden set: 30 answerable questions (26 single-paper, 4 multi-paper); unanswerable ones skipped
- Source: `.deepeval/.latest_test_run.json` (results copied from the saved run, not re-run)
- Run time ~10.9 s · judge cost ~$0.031

### Summary

| Metric | Mean | Passed (≥ 0.7) |
|---|---|---|
| Contextual Precision | **0.708** | 19 / 30 |
| Contextual Recall | **0.956** | 28 / 30 |

Test cases passing both metrics: 18 / 30 (12 failed).

| Question type | n | Precision | Recall |
|---|---|---|---|
| single_paper | 26 | 0.683 | 0.962 |
| multi_paper | 4 | 0.875 | 0.917 |

### Per-question scores

| ID | Type | Precision | Recall | Question |
|---|---|---|---|---|
| q01 | single | 0.25 | 1.00 | What retriever does the original RAG model use, and how is it initialized? |
| q02 | single | 0.95 | 1.00 | What is the difference between RAG-Sequence and RAG-Token? |
| q03 | single | 0.33 | **0.00** | Which components of RAG are updated during fine-tuning, and which are kept frozen? |
| q04 | single | 0.50 | 1.00 | What generator model and knowledge source does RAG combine? |
| q05 | single | 0.87 | 1.00 | How does Dense Passage Retrieval choose negative examples during training? |
| q06 | single | 1.00 | 1.00 | How much does DPR improve over BM25 on passage retrieval? |
| q07 | single | 0.92 | 1.00 | What is late interaction in ColBERT and how is the relevance score computed? |
| q08 | single | 1.00 | 1.00 | Why can ColBERT pre-compute document representations offline? |
| q09 | single | 0.89 | 1.00 | How does the position of relevant information in the input context affect LM performance? |
| q10 | single | 0.33 | 1.00 | Which tasks were used to study the lost-in-the-middle effect? |
| q11 | single | 0.92 | 1.00 | What metrics does RAGAS propose for evaluating RAG pipelines? |
| q12 | single | 0.80 | 1.00 | How does RAGAS compute the faithfulness score? |
| q13 | single | 0.64 | 1.00 | What are reflection tokens in Self-RAG? |
| q14 | single | 0.53 | 1.00 | How is Self-RAG trained? |
| q15 | single | 0.45 | 1.00 | How does HyDE build the query representation used for retrieval? |
| q16 | single | 0.20 | 1.00 | Why is HyDE robust to hallucinated details in the generated document? |
| q17 | single | 0.25 | 1.00 | What does Corrective RAG do when the retrieved documents are judged incorrect? |
| q18 | single | **0.00** | 1.00 | What is the decompose-then-recompose step in CRAG? |
| q19 | single | 1.00 | 1.00 | How does GraphRAG answer global questions about an entire corpus? |
| q20 | single | 0.81 | 1.00 | Why does naive vector RAG struggle with sensemaking questions? |
| q21 | single | 0.89 | 1.00 | How does ReAct combine reasoning and acting? |
| q22 | single | 0.81 | 1.00 | Why are dot products scaled in Transformer attention? |
| q23 | single | 1.00 | 1.00 | How does BERT's masked language modeling choose and replace tokens? |
| q24 | single | 0.92 | 1.00 | Why is plain BERT inefficient for semantic similarity search, and how does SBERT fix it? |
| q25 | single | 0.92 | 1.00 | What are the three RAG paradigms described in the RAG survey? |
| q26 | single | 0.58 | 1.00 | What is in-context learning as described in the GPT-3 paper? |
| q27 | multi | 0.83 | 1.00 | How do DPR and ColBERT differ in how they represent and match queries and passages? |
| q28 | multi | 1.00 | 0.67 | Self-RAG and CRAG both evaluate retrieved documents. How do their approaches differ? |
| q29 | multi | 0.75 | 1.00 | Which papers rely on DPR-style dense retrieval, and for what purpose? |
| q30 | multi | 0.92 | 1.00 | What do Lost in the Middle and the RAG survey suggest about passage count/order? |

### Observations

- **Recall is strong (0.96):** the right facts almost always appear somewhere in the top 5.
  - Misses: q03 (0.00, RAG fine-tuning / frozen components: no supporting chunk retrieved) and q28 (0.67, Self-RAG reflection-token side missing).
- **Precision is the weak spot (0.71, 11 failures):** relevant chunks are often ranked below irrelevant ones.
  - Worst: q18 (0.00), q16 (0.20), q01 / q17 (0.25), q03 / q10 (0.33).
  - Common pattern: generic "RAG" chunks from other papers (GraphRAG, CRAG related-work, the survey) outrank the target paper's chunk.
- Next ideas: add a reranker (cross-encoder), try hybrid BM25 + dense, or boost by paper title/section; then re-run and compare against this baseline.
