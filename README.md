# RAG Chatbot over AI/NLP Research Papers

Ask questions about 15 landmark papers (Transformers, BERT, DPR, ColBERT, RAG, Self-RAG,
CRAG, GraphRAG, RAGAS, …) and get answers with citations to the paper, section and page.

**Status:** Phase 1 — retrieval + generation pipeline with an evaluated baseline.

## How it works

```
PDFs ──► unstructured (partition + chunk by section) ──► BGE-small embeddings ──► Qdrant
                                                                                    │
question ──► embed ──► top-k chunks ──► prompt with numbered sources ──► gpt-4o-mini ──► cited answer
```

- **Parsing & chunking:** `unstructured` (`partition_pdf`, `chunk_by_title`), grouped by paper section;
  the References section, page headers and arXiv stamps are dropped.
- **Metadata on every chunk:** arXiv id, title, section, page range, document hash.
- **Idempotent ingestion:** unchanged PDFs are skipped (content hash); changed ones are replaced.
- **Grounded answers:** the model answers only from retrieved sources, cites them as `[n]`,
  and says "I don't know" when the papers don't cover the question.

## Stack

Python 3.12 · uv · unstructured · sentence-transformers (`BAAI/bge-small-en-v1.5`) · Qdrant ·
OpenAI (`gpt-4o-mini`) · DeepEval (LLM-as-judge evaluation)

## Setup

```bash
git clone <repo-url> && cd rag-chatbot
uv sync
cp .env.example .env                      # add your OPENAI_API_KEY
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant

uv run python scripts/download_papers.py  # 15 PDFs from arXiv → data/raw/
uv run python -m rag.ingestion.pipeline   # chunk, embed, load into Qdrant
```

## Usage

```bash
uv run python scripts/try_retrieval.py "What is late interaction in ColBERT?"   # inspect retrieved chunks
uv run python scripts/ask.py "How does Self-RAG decide when to retrieve?"        # cited answer
uv run python scripts/inspect_chunks.py 2005.11401.pdf                          # check chunking for one paper
```

## Evaluation

Golden set: `goldens/retrieval_golden.jsonl` — 33 questions (26 single-paper, 4 multi-paper,
3 unanswerable), each with relevant papers, a reference answer and the key facts a complete
answer needs.

```bash
uv run python eval/eval_retriever.py      # DeepEval contextual precision & recall, judge: gpt-4o-mini
```

**Baseline retriever** (dense only, BGE-small, top-k = 5, 30 answerable questions):

| Metric | Score | Passed (≥ 0.7) |
|---|---|---|
| Contextual recall | **0.956** | 28 / 30 |
| Contextual precision | **0.708** | 19 / 30 |

Recall is strong: the needed facts are almost always somewhere in the top 5. Precision is
the weak spot: relevant chunks are often ranked below irrelevant ones (e.g. CRAG and HyDE
questions). Hybrid search and reranking (Phase 3) target exactly this. Full per-question
results: [`eval/notes.md`](eval/notes.md).

## Project structure

```
src/rag/
  config.py            settings from .env
  embeddings.py        local (BGE) or OpenAI embeddings
  vectorstore.py       Qdrant: collection, upsert, search
  retrieval.py         question → ranked chunks with citations
  generation.py        prompt + LLM → cited answer
  prompts/answers.txt  answer prompt
  ingestion/
    chunker.py         PDF → section-aware chunks + metadata
    pipeline.py        PDFs → chunks → vectors → Qdrant
scripts/               download, ingest checks, retrieval and ask scripts
eval/                  evaluation script + baseline notes
goldens/               golden question set
```

## Roadmap

- [x] **Phase 1** — ingestion, retrieval, cited generation, evaluated baseline
- [ ] **Phase 2** — FastAPI backend with streaming answers
- [ ] **Phase 3** — LangGraph orchestration, hybrid search (dense + BM25), reranking, query rewriting
- [ ] **Phase 4** — Next.js chat UI, chat history (Postgres), auth, tracing (Langfuse)
- [ ] **Phase 5** — Docker Compose, CI with eval gates, guardrails, cloud deployment

> Answers come from the papers in `data/raw` only and may be incomplete. Not a substitute for reading the papers.
