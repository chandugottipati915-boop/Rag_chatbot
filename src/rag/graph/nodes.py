"""Graph nodes. Each node takes the state and returns ONLY the fields it changed."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from rag.config import settings
from rag.generation import NO_ANSWER, get_client
from rag.generation import generate as llm_generate
from rag.graph.state import RAGState
from rag.retrieval import retrieve as do_retrieve

log = logging.getLogger("rag.graph")

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
REWRITE_PROMPT = (PROMPTS / "rewrite.txt").read_text()
GRADE_PROMPT = (PROMPTS / "grade.txt").read_text()
REFORMULATE_PROMPT = (PROMPTS / "reformulate.txt").read_text()

MAX_HISTORY_MESSAGES = 4       # only the last few turns matter for resolving "it"
MAX_MESSAGE_CHARS = 600        # long answers in history are trimmed to keep the prompt small
MAX_PASSAGE_CHARS = 1200       # enough to judge relevance, keeps the grading call cheap


class Rewrite(BaseModel):
    query: str
    paper: str | None = None


class Grade(BaseModel):
    relevant: list[int]


class Reformulation(BaseModel):
    query: str


@lru_cache
def paper_catalog() -> dict[str, str]:
    """arXiv id -> title, from data/raw/catalog.json (empty if missing)."""
    try:
        data = json.loads((settings.raw_dir / "catalog.json").read_text())
        return {pid: meta.get("title", pid) for pid, meta in data.items()}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _format_history(history: list[dict]) -> str:
    lines = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        who = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {str(msg.get('content', ''))[:MAX_MESSAGE_CHARS]}")
    return "\n".join(lines)


# ------------------------------------------------------------------ nodes

def rewrite(state: RAGState) -> dict:
    """Turn a follow-up into a standalone query, and optionally pick one paper to search."""
    question = state["question"]
    history = state.get("history") or []

    if not history:                                   # nothing to resolve → no LLM call
        return {"query": question, "paper_filter": None}

    papers = paper_catalog()
    prompt = REWRITE_PROMPT.format(
        papers="\n".join(f"- {pid}: {title}" for pid, title in papers.items()) or "(unknown)",
        history=_format_history(history),
        question=question,
    )
    try:
        resp = get_client().chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        out = Rewrite.model_validate_json(resp.choices[0].message.content)
    except (OpenAIError, ValidationError, ValueError, TypeError):
        # never break the chat because of the rewrite: fall back to the raw question
        log.warning("Query rewrite failed; using the original question", exc_info=True)
        return {"query": question, "paper_filter": None}

    query = out.query.strip() or question
    paper = out.paper if out.paper in papers else None    # ignore invented ids
    if not settings.rewrite_paper_filter:
        paper = None                                       # filter disabled (see config)
    log.info("rewrite: %r -> %r (paper=%s)", question, query, paper)
    return {"query": query, "paper_filter": paper}


def retrieve(state: RAGState) -> dict:
    """Hybrid + rerank retrieval (per settings) with the standalone query."""
    query = state["query"]
    top_k = state.get("top_k") or settings.top_k
    forced = state.get("arxiv_id")                      # chosen by the user: always respected
    paper = forced or state.get("paper_filter")         # chosen by the rewrite: a best guess
    chunks = do_retrieve(query, top_k=top_k, arxiv_id=paper)
    if paper and not forced and not chunks:
        # a wrong guessed filter would hide the answer: retry across all papers
        log.info("No chunks in paper %s; retrying without the filter", paper)
        chunks = do_retrieve(query, top_k=top_k)
    return {"chunks": chunks}


def answer_question(state: RAGState) -> str:
    """The question the generator answers: the user's own words, plus how they were
    interpreted when rewritten (the generator doesn't see the chat history).
    Shared by the generate node and the streaming API."""
    question, query = state["question"], state.get("query", state["question"])
    return question if query == question else f"{question}\n(Interpreted as: {query})"


def generate(state: RAGState) -> dict:
    """Answer the user's own question from the graded chunks."""
    return {"answer": llm_generate(answer_question(state), state.get("chunks", []))}


def grade(state: RAGState) -> dict:
    """Keep only the chunks the LLM judges relevant; one call grades all chunks at once."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"retrieved": [], "chunks": []}

    passages = "\n\n".join(
        f"[{i}] ({c.title}, {c.section})\n{c.text[:MAX_PASSAGE_CHARS]}"
        for i, c in enumerate(chunks, 1)
    )
    # grade against the standalone query: "How is it trained?" can't be judged on its own
    prompt = GRADE_PROMPT.format(question=state.get("query", state["question"]), passages=passages)
    try:
        resp = get_client().chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        out = Grade.model_validate_json(resp.choices[0].message.content)
    except (OpenAIError, ValidationError, ValueError, TypeError):
        # grading is a filter: if it fails, keep everything (today's behavior), never lose context
        log.warning("Grading failed; keeping all %d chunks", len(chunks), exc_info=True)
        return {"retrieved": chunks, "chunks": chunks}

    keep = {n for n in out.relevant if 1 <= n <= len(chunks)}   # passages are numbered from 1
    kept = [c for i, c in enumerate(chunks, 1) if i in keep]     # original (reranked) order
    log.info("grade: kept %d/%d chunks %s", len(kept), len(chunks), sorted(keep))
    return {"retrieved": chunks, "chunks": kept}


def abstain(state: RAGState) -> dict:
    """Nothing relevant was found: refuse without calling the generator."""
    return {"answer": NO_ANSWER}


def reformulate(state: RAGState) -> dict:
    """Nothing relevant was found: write a different search query and try once more.

    Search-oriented rewriting (Rewrite-Retrieve-Read), used as a recovery step. The paper
    filter is dropped too, in case it was what hid the answer.
    """
    question = state["question"]
    failed = state.get("query", question)
    retries = state.get("retries", 0) + 1
    prompt = REFORMULATE_PROMPT.format(
        papers="\n".join(f"- {pid}: {title}" for pid, title in paper_catalog().items()) or "(unknown)",
        question=question,
        failed_query=failed,
    )
    try:
        resp = get_client().chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        query = Reformulation.model_validate_json(resp.choices[0].message.content).query.strip()
    except (OpenAIError, ValidationError, ValueError, TypeError):
        log.warning("Reformulation failed; retrying with the original question", exc_info=True)
        query = question
    query = query or question
    log.info("reformulate (retry %d): %r -> %r", retries, failed, query)
    return {"query": query, "paper_filter": None, "retries": retries}


def route_after_grade(state: RAGState) -> Literal["generate", "reformulate", "abstain"]:
    """Conditional edge (not a node): returns the name of the next node."""
    if state.get("chunks"):
        return "generate"
    if state.get("retries", 0) < settings.max_retries:
        return "reformulate"                      # one more search with a different query
    return "abstain"

