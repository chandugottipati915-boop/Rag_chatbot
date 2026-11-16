"""Graph nodes. Each node takes the state and returns ONLY the fields it changed."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from rag.config import settings
from rag.generation import generate as llm_generate
from rag.generation import get_client
from rag.graph.state import RAGState
from rag.retrieval import retrieve as do_retrieve

log = logging.getLogger("rag.graph")

REWRITE_PROMPT = (Path(__file__).resolve().parents[1] / "prompts" / "rewrite.txt").read_text()
MAX_HISTORY_MESSAGES = 4       # only the last few turns matter for resolving "it"
MAX_MESSAGE_CHARS = 600        # long answers in history are trimmed to keep the prompt small


class Rewrite(BaseModel):
    query: str
    paper: str | None = None


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
    log.info("rewrite: %r -> %r (paper=%s)", question, query, paper)
    return {"query": query, "paper_filter": paper}


def retrieve(state: RAGState) -> dict:
    """Hybrid + rerank retrieval (per settings) with the standalone query."""
    query, paper = state["query"], state.get("paper_filter")
    chunks = do_retrieve(query, top_k=settings.top_k, arxiv_id=paper)
    if paper and not chunks:
        # a wrong paper filter would hide the answer: retry across all papers
        log.info("No chunks in paper %s; retrying without the filter", paper)
        chunks = do_retrieve(query, top_k=settings.top_k)
    return {"chunks": chunks}


def generate(state: RAGState) -> dict:
    """Answer the user's own question; mention how it was interpreted if it was rewritten."""
    question, query = state["question"], state.get("query", state["question"])
    if query != question:
        # the generator doesn't see the history, so tell it what "it" referred to
        question = f"{question}\n(Interpreted as: {query})"
    return {"answer": llm_generate(question, state.get("chunks", []))}
