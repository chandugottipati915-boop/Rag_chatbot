from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from rag.config import settings
from rag.retrieval import RetrievedChunk, retrieve

PROMPT = (Path(__file__).parent / "prompts" / "answers.txt").read_text()
NO_ANSWER = "I don't know based on the provided papers."


@lru_cache
def get_client() -> OpenAI:
    # created on first use, so importing this module doesn't need an API key (tests rely on this)
    return OpenAI(api_key=settings.openai_api_key)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]

    def format(self) -> str:
        lines = [self.text, "", "Sources:"]
        lines += [f"  [{i}] {s.citation}" for i, s in enumerate(self.sources, 1)]
        return "\n".join(lines)


def build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.title}, {c.section}, p. {c.page_start})\n{c.text}"
        for i, c in enumerate(chunks, 1)
    )


def _messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    return [{"role": "user", "content": PROMPT.format(
        context=build_context(chunks), question=question,
    )}]


def generate(question: str, chunks: list[RetrievedChunk]) -> str:
    """LLM answer over chunks you already retrieved."""
    if not chunks:
        return NO_ANSWER
    resp = get_client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=_messages(question, chunks),
    )
    return resp.choices[0].message.content.strip()


def generate_stream(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """Like generate(), but yields pieces of text as the LLM writes them."""
    if not chunks:
        yield NO_ANSWER
        return
    stream = get_client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=_messages(question, chunks),
        stream=True,
    )
    for event in stream:
        if event.choices and (piece := event.choices[0].delta.content):
            yield piece


def answer(question: str, top_k: int = settings.top_k) -> Answer:
    """Retrieve + generate in one call (used by scripts/ask.py)."""
    chunks = retrieve(question, top_k=top_k)
    return Answer(question, generate(question, chunks), chunks)
