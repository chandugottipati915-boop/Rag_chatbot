# RAG Chatbot over AI/NLP Research Papers

Ask questions about 15 landmark papers (Transformers, BERT, DPR, ColBERT, RAG, Self-RAG,
CRAG, GraphRAG, RAGAS, …) and get answers with citations to the paper, section and page.
Follow-up questions work: the conversation is taken into account.

**Status:** Phase 4 in progress. Hybrid retrieval with reranking, orchestrated as a LangGraph
pipeline (query rewriting, relevance grading, one retry), served by a streaming FastAPI backend,
measured with three evaluation sets, and used through a Next.js chat UI.

![The chat UI: an answer with numbered citations, and the cited passage highlighted in the margin](docs/ui.png)

## How it works

**Indexing**

```
PDFs ──► unstructured (partition + chunk by section) ──► OpenAI embeddings ──┐
                                                          BM25 (in Qdrant) ──┴──► Qdrant
```

**Answering a question** (LangGraph)

```
question + history
   │
   ▼
rewrite ──► retrieve ──► grade ──┬── relevant chunks ─────────────► generate ──► cited answer
              ▲                  ├── nothing relevant, retry left ─► reformulate ─┐
              └──────────────────┼────────────────────────────────────────────────┘
                                 └── nothing relevant, no retry ───► abstain ───► "I don't know…"
```

| Step | What it does | Why |
|---|---|---|
| **rewrite** | Turns a follow-up ("How is it trained?") into a standalone query ("How is Self-RAG trained?") using the last turns. No LLM call when there is no history. | Retrieval only sees one message; without this, follow-ups retrieve the wrong papers. |
| **retrieve** | Hybrid search (dense + BM25, fused with RRF), 20 candidates reranked by a cross-encoder to the top 5. | Dense search misses rare terms (CRAG, MaxSim); the reranker fixes the ordering. |
| **grade** | One LLM call keeps only the chunks that actually help answer the question. | Stops wrong-paper chunks (e.g. CRAG's appendix mentioning Self-RAG) from reaching the answer. |
| **reformulate** | If nothing is relevant, writes one different search query and retrieves again (max 1 retry). | Gives missed questions a second chance at bounded cost. |
| **generate / abstain** | Answers from the graded chunks with `[n]` citations, corrects false premises, or refuses without an LLM call. | Grounded answers; honest refusals when the papers don't cover the question. |

Every LLM step falls back safely: a failed rewrite uses the original question, a failed
grade keeps all chunks, a failed reformulation retries with the original question.

## Results

Three golden sets: **retrieval** (33 questions), **generator** (43 questions incl. false premises
and unanswerable ones) and **multi-turn** (14 conversations). Judge: `gpt-4o-mini` via DeepEval.

**Retrieval** (30 answerable questions)

| Setup | Contextual precision | Contextual recall | Retrieval p50 |
|---|---|---|---|
| Baseline: BGE dense | 0.708 | 0.956 | — |
| OpenAI hybrid (dense + BM25) | 0.838 | 0.936 | — |
| **OpenAI hybrid + reranker** | **0.881** | **0.969** | 1.3 s |

**Generator** (33 questions that should be answered)

| Setup | Faithfulness | Answer relevancy | Correct behavior | False refusals |
|---|---|---|---|---|
| Baseline (first prompt) | 0.899 | 0.843 | — | 5 |
| Hybrid + rerank, new prompt | 0.902 | 0.966 | 0.715 | 1 |
| **Full graph** | **0.928** | 0.948 | 0.703 | 1 |

Unanswerable questions are refused 4/4 in every setup since the prompt fix. Through the graph,
2 of 43 questions needed the retry step; grading lifts faithfulness (0.902 → 0.928) because
irrelevant chunks no longer reach the generator.

**Multi-turn** (14 conversations; the same follow-up sent with and without the earlier turns)

| | Paper hit | Recall | Correct behavior | Rewrite quality |
|---|---|---|---|---|
| Without history | 0.62 | 0.54 | 0.54 | — |
| **With history (rewriting)** | **1.00** | **0.96** | **0.82** | 0.90 |

**Latency** (full graph on a laptop)

| Questions | p50 | p95 |
|---|---|---|
| Paper questions (33) | 4.4 s | 5.7 s |
| Off-topic (grade → retry → abstain) | 5.5 s | 6.0 s |

Retrieval with reranking takes ~1.3 s of this; grading and generation are `gpt-4o-mini` calls.
`/chat/stream` sends the sources as soon as grading finishes and then streams the answer,
so users see output well before the full time. Off-topic questions are the slowest because
they pay for the retry before being refused (a router step could skip this).

**Known limitations**
- False premises in single questions are corrected only 1 in 4 times (`gpt-4o-mini` tends to refuse).
- Ambiguous questions ("How is the relevance score computed?") are not clarified.
- Comparisons sometimes retrieve only one side (DPR vs ColBERT).
- One detail-level miss (RAG's "21 million documents") in every setup.

Details and per-question notes: [`eval/phase3_comparison.md`](eval/phase3_comparison.md),
[`eval/retrieval_eval_notes.md`](eval/retrieval_eval_notes.md),
[`eval/generator_eval_notes.md`](eval/generator_eval_notes.md).

### Design decisions backed by the evals
- **Hybrid + reranker as default:** best precision and recall of all setups, for ~1.3 s per query.
- **No paper filter from the rewrite:** it looked useful, but in the multi-turn eval a wrong guess
  hid the answer twice and hybrid + rerank found the right paper without it in every case.
  It's still available behind `REWRITE_PAPER_FILTER=true`.
- **One retry, not more:** enough to recover missed questions; each retry adds an LLM call and
  a search, which off-topic questions pay for before being refused.

## Stack

Python 3.12 · uv · unstructured · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · Qdrant
(dense + BM25, RRF) · sentence-transformers cross-encoder (`BAAI/bge-reranker-base`) ·
LangGraph · FastAPI · Postgres (SQLAlchemy 2, Alembic) · DeepEval · pytest · Next.js 16 (React 19, TypeScript, Tailwind v4) · Vitest

## Setup

```bash
git clone <repo-url> && cd rag-chatbot
uv sync
cp .env.example .env                      # add OPENAI_API_KEY; set RETRIEVAL_MODE=hybrid, RERANK=true
docker run -d -p 6333:6333 -v qdrant_data:/qdrant/storage --name qdrant qdrant/qdrant   # Qdrant >= 1.15.3
docker run -d -p 5432:5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -v pg_data:/var/lib/postgresql/data --name postgres postgres:17                       # chat history
uv run alembic upgrade head               # create the chat-history tables

uv run python scripts/download_papers.py  # 15 PDFs from arXiv → data/raw/
uv run python scripts/ingest.py           # chunk, embed, load into Qdrant (--force to rebuild)
```

## Usage

```bash
uv run python scripts/try_graph.py                    # run the full graph: a question, a follow-up, edge cases
uv run python scripts/try_graph.py --diagram          # print the graph as Mermaid
uv run python scripts/try_retrieval.py "What is late interaction in ColBERT?"   # inspect retrieved chunks
uv run python scripts/inspect_chunks.py 2005.11401.pdf                          # check chunking for one paper
```

## Chat UI

```bash
cd frontend && npm install && npm run dev          # http://localhost:3000 (the API must be running)
```

A Next.js app that streams answers from `/chat/stream`. Each `[n]` citation is a clickable
mark that opens the cited passage in a side margin, and chats are saved in Postgres through
the API, so they survive reloads and other browsers. Details in [frontend/README.md](frontend/README.md).

## API

```bash
uv run uvicorn rag.api.main:app --app-dir src --reload     # interactive docs: http://localhost:8000/docs
```

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Qdrant reachable + number of indexed chunks (`ok` / `degraded`), and whether the database is up |
| GET | `/papers` | The indexed papers (for the UI's paper filter) |
| POST | `/search` | Retrieved chunks only, no LLM call |
| POST | `/chat` | Full cited answer as JSON (runs the graph) |
| POST | `/chat/stream` | Streamed answer via Server-Sent Events: `sources` → `token`… → `done` |
| POST / GET | `/ingest` | Start background ingestion / check its status |
| GET / POST | `/conversations` | List saved chats (newest first) / start one |
| GET / PATCH / DELETE | `/conversations/{id}` | A chat with its messages / rename / delete |

For a saved chat, send `conversation_id`: the server loads the history from Postgres and stores
both turns (a stopped answer is kept, marked `stopped`). Without it, send earlier turns in `history`
yourself. Responses include `rewritten_query` and `retries`:

```bash
curl -s -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{
  "question": "How is it trained?",
  "history": [{"role": "user", "content": "What is Self-RAG?"},
              {"role": "assistant", "content": "Self-RAG trains a model to decide when to retrieve..."}]
}' | python -m json.tool
```

Errors: `404` unknown conversation · `422` invalid input · `502` LLM provider failed · `503` vector store unavailable · `409` ingestion already running.

## Testing and evaluation

```bash
uv run pytest -q                                   # 50 tests: API + graph, all external services faked
(cd frontend && npm test)                          # 10 UI unit tests: SSE parser, citations, saved chats
uv run python eval/eval_retriever.py               # contextual precision & recall
uv run python eval/eval_generator.py               # faithfulness, relevancy, correct behavior (through the graph)
uv run python eval/eval_multiturn.py               # follow-ups, with vs without history
uv run python scripts/bench_latency.py --graph     # full-graph latency
bash scripts/run_eval.sh hybrid+rerank             # ingest + latency + retrieval & generator evals for a setup
```

## Project structure

```
src/rag/
  config.py            settings from .env (retrieval mode, reranker, retries, ...)
  embeddings.py        OpenAI or local (BGE) embeddings
  vectorstore.py       Qdrant: dense or hybrid (dense + BM25) collections and search
  rerank.py            cross-encoder reranker
  retrieval.py         question → ranked chunks with citations
  generation.py        prompts + LLM → cited answer (also streaming)
  prompts/             system rules, answer template, rewrite, grade, reformulate
  graph/
    state.py           shared graph state
    nodes.py           rewrite, retrieve, grade, reformulate, generate, abstain
    build.py           graph wiring (full graph + retrieval-only graph for streaming)
  db/
    models.py          conversations + messages tables (SQLAlchemy)
    session.py         engine and per-request session
    repo.py            chat-history queries
  api/
    main.py            FastAPI routes
    schemas.py         request / response models
  ingestion/
    chunker.py         PDF → section-aware chunks + metadata
    pipeline.py        PDFs → chunks → vectors → Qdrant
scripts/               download, ingest, try the graph, retrieval checks, latency, eval runner
eval/                  three eval scripts + notes and comparison tables
goldens/               retrieval, generator and multi-turn golden sets
tests/                 API, graph and chat-history tests
migrations/            Alembic migrations for the chat-history tables
frontend/              Next.js chat UI (see frontend/README.md)
  app/                 layout, page, global styles
  components/          Chat, Answer, SourceList, Composer, Sidebar, ...
  lib/                 API client, SSE parser, citation linking, storage
```

## Roadmap

- [x] **Phase 1**: ingestion, retrieval, cited generation, evaluated baseline
- [x] **Phase 2**: FastAPI backend with streaming answers
- [x] **Phase 3**: hybrid search + reranking, LangGraph (rewrite, grade, retry), multi-turn chat, three eval sets
- [ ] **Phase 4**: Next.js chat UI, chat history (Postgres), auth, tracing (Langfuse)
  - [x] Chat UI: streaming answers, clickable citations, paper filter, browser-saved history
  - [x] Server-side chat history (Postgres, Alembic migrations)
  - [ ] Auth
  - [ ] Tracing (Langfuse)
- [ ] **Phase 5**: Docker Compose, CI with eval gates, guardrails, cloud deployment

> Answers come from the papers in `data/raw` only and may be incomplete. Not a substitute for reading the papers.
