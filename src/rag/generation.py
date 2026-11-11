from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from rag.config import settings
from rag.retrieval import RetrievedChunk, retrieve

PROMPT = (Path(__file__).parent / "prompts" / "answers.txt").read_text()
client = OpenAI(api_key=settings.openai_api_key)


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


def answer(question: str, top_k: int = settings.top_k) -> Answer:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return Answer(question, "I don't know based on the provided papers.", [])

    resp = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        messages=[{"role": "user", "content": PROMPT.format(
            context=build_context(chunks), question=question,
        )}],
    )
    return Answer(question, resp.choices[0].message.content.strip(), chunks)