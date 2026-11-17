"""Tracing helpers: off by default (see conftest.py), and the metadata LangSmith groups by."""
import uuid

from rag.api.schemas import ChatRequest
from rag.tracing import end_turn, start_turn, turn_metadata, under


def test_tracing_is_off_in_tests_so_nothing_is_sent():
    req = ChatRequest(question="What is RAG?")
    turn = start_turn("chat", req)
    assert turn is None
    with under(turn):          # no-ops when tracing is off
        pass
    end_turn(turn, outputs={"answer": "x"})


def test_metadata_groups_turns_by_conversation():
    cid = uuid.uuid4()
    meta = turn_metadata(ChatRequest(question="q", conversation_id=cid, arxiv_id="2310.11511"))
    assert meta["conversation_id"] == str(cid)      # LangSmith shows a chat's turns as one thread
    assert meta["arxiv_id"] == "2310.11511"
    assert {"retrieval_mode", "rerank", "llm_model"} <= meta.keys()
    assert "conversation_id" not in turn_metadata(ChatRequest(question="q"))
