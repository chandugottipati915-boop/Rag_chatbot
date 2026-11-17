import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

import argparse

from rag.config import PROJECT_ROOT
from rag.generation import NO_ANSWER, generate
from rag.graph.build import get_graph
from rag.retrieval import retrieve

THRESHOLD = 0.7
JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 5
GOLDEN = PROJECT_ROOT / "goldens" / "generator_golden.jsonl"
OUT_DIR = PROJECT_ROOT / "eval" / "results"

golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

# graph (default): the same pipeline as the API: rewrite → retrieve → grade → retry → generate/abstain
# direct:          the older pipeline: retrieve → generate, kept for comparison with earlier runs
ap = argparse.ArgumentParser()
ap.add_argument("--pipeline", choices=["graph", "direct"], default="graph")
PIPELINE = ap.parse_args().pipeline


def run_pipeline(question: str) -> tuple[str, list, dict]:
    """Return (answer, chunks the answer was based on, extra info)."""
    if PIPELINE == "direct":
        chunks = retrieve(question, top_k=TOP_K)
        return generate(question, chunks), chunks, {}
    state = get_graph().invoke({"question": question})
    chunks = state.get("chunks", [])
    # when the graph abstains, no chunks survive grading: judge the refusal against what was
    # retrieved, so Faithfulness still has context (a refusal makes no claims anyway)
    context = chunks or state.get("retrieved", [])
    return state["answer"], context, {"retries": state.get("retries", 0), "kept": len(chunks),
                                      "retrieved": len(state.get("retrieved", []))}

# What the judge should expect, per `expected_behavior` in the golden set
BEHAVIOR_SPEC = {
    "answer": "ANSWER. The papers contain the answer. The response should answer the question. "
              "Refusing ('I don't know') is wrong.",
    "correct_premise": "CORRECT THE PREMISE. The question contains a false assumption. The response must "
                       "say the assumption is wrong and give the correct information. Refusing, or "
                       "accepting the false assumption, is wrong.",
    "abstain": "DECLINE. The papers do not contain the answer. The response must decline (for example "
               "'I don't know based on the provided papers') and must not answer from general knowledge "
               "or invent facts.",
    "clarify_or_cover": "CLARIFY OR COVER. The question is ambiguous. The response should ask which meaning "
                        "is intended, or explicitly cover the possible meanings. Silently picking one "
                        "meaning is wrong.",
}

# ---- 1. run the pipeline
print(f"Pipeline: {PIPELINE}")
test_cases, records = [], []
for i, item in enumerate(golden, 1):
    print(f"[{i}/{len(golden)}] {item['id']} {item['question'][:60]}")
    answer, chunks, info = run_pipeline(item["question"])

    test_cases.append(
        LLMTestCase(
            input=item["question"],
            actual_output=answer,                               # what the generator actually said
            # read by the behavior metric (GEval); Faithfulness and Answer Relevancy ignore it
            expected_output=(f"Expected behavior: {BEHAVIOR_SPEC[item['expected_behavior']]}\n"
                             f"Reference answer: {item['reference_answer']}"),
            retrieval_context=[c.text for c in chunks] or ["(no sources retrieved)"],  # Faithfulness context
            additional_metadata={
                "id": item["id"],
                "category": item["category"],
                "expected_behavior": item["expected_behavior"],
            },
        )
    )
    records.append({
        "id": item["id"], "category": item["category"],
        "expected_behavior": item["expected_behavior"],
        "question": item["question"], "answer": answer,
        "sources": [c.citation for c in chunks],
        "pipeline": PIPELINE, **info,
    })

# save the generated answers: they cost money, and you'll want to read them
OUT_DIR.mkdir(parents=True, exist_ok=True)
answers_file = OUT_DIR / f"generator_answers_{PIPELINE}_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
answers_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
print(f"\nSaved answers → {answers_file}")

# ---- 2. judge them
behavior_metric = GEval(
    name="Correct Behavior",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                       SingleTurnParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Read the 'Expected behavior' line in the expected output: ANSWER, CORRECT THE PREMISE, "
        "DECLINE, or CLARIFY OR COVER.",
        "Decide which of these behaviors the actual output shows.",
        "If the actual output shows a different behavior than expected, give a very low score.",
        "If the behavior matches and is ANSWER or CORRECT THE PREMISE, check the content against the "
        "reference answer: lower the score for contradictions, wrong facts or missing key points.",
        "If the behavior matches and is DECLINE or CLARIFY OR COVER, do not penalize the response "
        "for being short or for not containing facts.",
    ],
    model=JUDGE_MODEL,
    threshold=THRESHOLD,
)

metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    behavior_metric,
]
result = evaluate(test_cases=test_cases, metrics=metrics)

# ---- 3. summary, grouped by expected behavior
# match results back by question (evaluate() runs cases concurrently, so don't rely on order)
behavior_of = {item["question"]: item["expected_behavior"] for item in golden}
scores = defaultdict(lambda: defaultdict(list))   # behavior → metric → [scores]
for tr in result.test_results:
    behavior = behavior_of[tr.input]
    for md in tr.metrics_data or []:
        scores[behavior][md.name].append(md.score or 0.0)

print("\n=== Generator eval (pipeline: %s, judge: %s, k=%d) ===" % (PIPELINE, JUDGE_MODEL, TOP_K))
for behavior in ("answer", "correct_premise", "abstain", "clarify_or_cover"):
    if behavior not in scores:
        continue
    print(f"\n{behavior}  (n={len(next(iter(scores[behavior].values())))})")
    for name, vals in scores[behavior].items():
        passed = sum(v >= THRESHOLD for v in vals)
        print(f"  {name:<20} mean={sum(vals)/len(vals):.3f}  passed={passed}/{len(vals)}")

# deterministic check, no judge needed: answerable questions that got the refusal sentence
false_refusals = [r["id"] for r in records
                  if r["expected_behavior"] == "answer" and r["answer"].strip() == NO_ANSWER]
missed_abstains = [r["id"] for r in records
                   if r["expected_behavior"] == "abstain" and r["answer"].strip() != NO_ANSWER]
print(f"\nFalse refusals on answerable questions: {len(false_refusals)}  {false_refusals}")
print(f"Unanswerable questions NOT refused:     {len(missed_abstains)}  {missed_abstains}")
if PIPELINE == "graph":
    retried = [r["id"] for r in records if r.get("retries")]
    print(f"Questions that needed a retry:          {len(retried)}  {retried}")
