"""LangGraph pipeline tests. The LLM, retrieval and generation are faked, so no
Qdrant, models or API key are needed.

Run: uv run pytest tests/test_graph.py -v
"""

import json
from types import SimpleNamespace

import pytest
from openai import OpenAIError

import rag.graph.nodes as nodes
from rag.generation import NO_ANSWER
from rag.graph.build import get_graph
from rag.retrieval import RetrievedChunk


def chunk(n: int, arxiv_id: str = "2310.11511") -> RetrievedChunk:
    return RetrievedChunk(f"text {n}", 0.9, arxiv_id, "Self-RAG", f"{n} Section", n, n)


class FakeLLM:
    """Returns the queued JSON replies in order; an Exception in the queue is raised.

    Routing calls for first messages are answered with `self.route` without using the
    queue, so tests about search and grading don't need to queue them."""

    def __init__(self, *replies, route="papers"):
        self.replies, self.prompts, self.route = list(replies), [], route
        self.chat = self.completions = self

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if prompt.startswith("Classify the user's latest message"):
            self.routed = True
            return self._reply({"route": self.route} if not isinstance(self.route, Exception) else self.route)
        self.prompts.append(prompt)
        return self._reply(self.replies.pop(0))

    def _reply(self, reply):
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake(monkeypatch):
    state = SimpleNamespace(chunks=[chunk(1), chunk(2), chunk(3, "2401.15884")],
                            generated_with=None, retrieve_calls=[])

    def fake_retrieve(query, top_k=5, arxiv_id=None):
        state.retrieve_calls.append((query, arxiv_id))
        return list(state.chunks)

    def fake_generate(question, chunks):
        state.generated_with = (question, chunks)
        return f"answer from {len(chunks)} chunks [1]"

    monkeypatch.setattr(nodes, "do_retrieve", fake_retrieve)
    monkeypatch.setattr(nodes, "llm_generate", fake_generate)
    monkeypatch.setattr(nodes, "paper_catalog", lambda: {"2310.11511": "Self-RAG", "2401.15884": "CRAG"})

    def use_llm(*replies, route="papers"):
        llm = FakeLLM(*replies, route=route)
        monkeypatch.setattr(nodes, "get_client", lambda: llm)
        return llm

    state.use_llm = use_llm
    return state


def run(question, history=None):
    inp = {"question": question}
    if history:
        inp["history"] = history
    return get_graph().invoke(inp)


# ------------------------------------------------------------------ grade + routing

def test_relevant_chunks_are_kept_in_order_and_generated(fake):
    fake.use_llm({"relevant": [3, 1]})                     # grade call only (no history → no rewrite)
    out = run("Which base models does Self-RAG fine-tune?")
    assert [c.section for c in out["chunks"]] == ["1 Section", "3 Section"]   # reranked order kept
    assert len(out["retrieved"]) == 3
    assert fake.generated_with is not None and len(fake.generated_with[1]) == 2


def test_wrong_paper_chunk_is_dropped(fake):
    fake.use_llm({"relevant": [1, 2]})                     # grader rejects the CRAG chunk (#3)
    out = run("Which base models does Self-RAG fine-tune?")
    assert all(c.arxiv_id == "2310.11511" for c in out["chunks"])


def test_nothing_relevant_retries_once_then_abstains(fake):
    llm = fake.use_llm({"relevant": []},                              # grade #1
                       {"query": "minutes per hour time conversion"},  # reformulate
                       {"relevant": []})                               # grade #2
    out = run("How many minutes are in an hour?")
    assert out["answer"] == NO_ANSWER
    assert out["retries"] == 1
    assert len(fake.retrieve_calls) == 2                   # exactly one extra search
    assert fake.generated_with is None                     # no generation call
    assert llm.replies == []                               # no extra LLM calls beyond the three


def test_retry_finds_relevant_chunks_and_generates(fake):
    fake.use_llm({"relevant": []},                                     # grade #1: nothing
                 {"query": "Self-RAG critic and generator training"},  # reformulate
                 {"relevant": [1]})                                    # grade #2: found it
    out = run("How do they teach the model?")
    assert fake.retrieve_calls[1] == ("Self-RAG critic and generator training", None)
    assert out["retries"] == 1
    assert fake.generated_with is not None and len(fake.generated_with[1]) == 1


def test_retry_drops_the_paper_filter(fake, monkeypatch):
    monkeypatch.setattr(nodes.settings, "rewrite_paper_filter", True)
    fake.use_llm({"query": "How is Self-RAG trained?", "paper": "2310.11511"},   # rewrite
                 {"relevant": []},                                              # grade #1
                 {"query": "Self-RAG training procedure"},                      # reformulate
                 {"relevant": [2]})                                             # grade #2
    run("How is it trained?", HISTORY)
    assert fake.retrieve_calls[0][1] == "2310.11511"
    assert fake.retrieve_calls[1] == ("Self-RAG training procedure", None)


def test_reformulation_failure_retries_with_original_question(fake):
    fake.use_llm({"relevant": []}, OpenAIError("down"), {"relevant": []})
    out = run("Some question")
    assert fake.retrieve_calls[1] == ("Some question", None)
    assert out["answer"] == NO_ANSWER


def test_empty_retrieval_skips_grading_but_still_retries(fake):
    fake.chunks = []
    llm = fake.use_llm({"query": "a different query"})    # only the reformulate call
    out = run("Anything")
    assert out["answer"] == NO_ANSWER
    assert len(llm.prompts) == 1 and "found nothing relevant" in llm.prompts[0]


def test_max_retries_zero_abstains_immediately(fake, monkeypatch):
    monkeypatch.setattr(nodes.settings, "max_retries", 0)
    fake.use_llm({"relevant": []})
    out = run("How many minutes are in an hour?")
    assert out["answer"] == NO_ANSWER and len(fake.retrieve_calls) == 1


@pytest.mark.parametrize("bad", [OpenAIError("down"), "not json", {"wrong_key": [1]}])
def test_grading_failure_keeps_all_chunks(fake, bad):
    fake.use_llm(bad)
    out = run("What is Self-RAG?")
    assert len(out["chunks"]) == 3                         # fallback = behavior before grading existed
    assert fake.generated_with is not None


def test_out_of_range_passage_numbers_are_ignored(fake):
    fake.use_llm({"relevant": [0, 2, 7, -1]})
    out = run("What is Self-RAG?")
    assert [c.section for c in out["chunks"]] == ["2 Section"]


def test_grader_sees_numbered_passages_and_the_query(fake):
    llm = fake.use_llm({"relevant": [1]})
    run("What is Self-RAG?")
    prompt = llm.prompts[-1]
    assert "Question: What is Self-RAG?" in prompt
    assert "[1] (Self-RAG, 1 Section)" in prompt and "[3]" in prompt


# ------------------------------------------------------------------ rewrite (C1) still works

HISTORY = [{"role": "user", "content": "What is Self-RAG?"},
           {"role": "assistant", "content": "Self-RAG is ..."}]


def test_follow_up_is_rewritten_and_grader_uses_the_rewritten_query(fake, monkeypatch):
    monkeypatch.setattr(nodes.settings, "rewrite_paper_filter", True)
    llm = fake.use_llm({"query": "How is Self-RAG trained?", "paper": "2310.11511"},   # rewrite
                       {"relevant": [1]})                                              # grade
    out = run("How is it trained?", HISTORY)
    assert out["query"] == "How is Self-RAG trained?"
    assert fake.retrieve_calls[0] == ("How is Self-RAG trained?", "2310.11511")
    assert "Question: How is Self-RAG trained?" in llm.prompts[-1]
    assert "(Interpreted as: How is Self-RAG trained?)" in fake.generated_with[0]


def test_invented_paper_id_is_ignored(fake, monkeypatch):
    monkeypatch.setattr(nodes.settings, "rewrite_paper_filter", True)
    fake.use_llm({"query": "How is Self-RAG trained?", "paper": "9999.99999"}, {"relevant": [1]})
    out = run("How is it trained?", HISTORY)
    assert out["paper_filter"] is None


def test_graph_shape():
    mermaid = get_graph().get_graph().draw_mermaid()
    for edge in ("retrieve --> grade", "grade -.-> generate", "grade -.-> abstain",
                 "grade -.-> reformulate", "reformulate --> retrieve"):
        assert edge in mermaid, edge


def test_paper_filter_is_off_by_default(fake):
    fake.use_llm({"query": "How is Self-RAG trained?", "paper": "2310.11511"}, {"relevant": [1]})
    out = run("How is it trained?", HISTORY)
    assert out["query"] == "How is Self-RAG trained?"      # the rewrite still works
    assert out["paper_filter"] is None                     # but no paper restriction
    assert fake.retrieve_calls[0] == ("How is Self-RAG trained?", None)



# ------------------------------------------------------------------ routing

def test_greeting_skips_search_and_gets_a_small_talk_reply(fake):
    llm = fake.use_llm("Hi! Ask me about the RAG papers.", route="chat")   # small-talk reply
    out = run("hi")
    assert out["answer"] == "Hi! Ask me about the RAG papers."
    assert fake.retrieve_calls == [] and fake.generated_with is None
    assert "is small talk, not a research question: hi" in llm.prompts[0]


def test_small_talk_failure_uses_the_default_reply(fake):
    fake.use_llm(OpenAIError("down"), route="chat")
    assert run("thanks!")["answer"] == nodes.SMALL_TALK_FALLBACK


def test_off_topic_is_refused_without_searching(fake):
    llm = fake.use_llm(route="off_topic")
    out = run("What is a good pancake recipe?")
    assert out["answer"] == NO_ANSWER
    assert fake.retrieve_calls == [] and llm.prompts == []         # no search, no grading


def test_routing_failure_falls_back_to_searching(fake):
    fake.use_llm({"relevant": [1]}, route=OpenAIError("down"))
    out = run("What is Self-RAG?")
    assert len(fake.retrieve_calls) == 1 and fake.generated_with is not None


def test_follow_up_small_talk_is_not_turned_into_a_research_question(fake):
    # the rewrite call routes follow-ups too: no separate routing call
    llm = fake.use_llm({"query": "How is Self-RAG trained?", "paper": None, "route": "chat"},
                       "You're welcome!")
    out = run("thanks!", HISTORY)
    assert out["answer"] == "You're welcome!"
    assert out["query"] == "thanks!" and fake.retrieve_calls == []
    assert not getattr(llm, "routed", False)


def test_retry_cannot_change_the_topic(fake):
    """The bug behind "hi" → HyDE: a retry's reformulated query must not decide relevance."""
    llm = fake.use_llm({"relevant": []},                                # grade #1 (Self-RAG question)
                       {"query": "HyDE hypothetical document embeddings"},  # a drifting reformulation
                       {"relevant": []})                                 # grade #2
    out = run("How does Self-RAG critique its outputs?")
    assert fake.retrieve_calls[1][0] == "HyDE hypothetical document embeddings"   # searched with it...
    assert "Question: How does Self-RAG critique its outputs?" in llm.prompts[-1]  # ...but graded on the user's question
    assert out["answer"] == NO_ANSWER


def test_generator_sees_the_standalone_question_not_the_retry_query(fake):
    fake.use_llm({"relevant": []}, {"query": "Self-RAG reflection token training"}, {"relevant": [1]})
    run("How does Self-RAG learn to critique?")
    assert fake.generated_with[0] == "How does Self-RAG learn to critique?"


def test_graph_has_the_router_edges():
    mermaid = get_graph().get_graph().draw_mermaid()
    for edge in ("rewrite -.-> retrieve", "rewrite -.-> small_talk", "rewrite -.-> abstain"):
        assert edge in mermaid, edge
