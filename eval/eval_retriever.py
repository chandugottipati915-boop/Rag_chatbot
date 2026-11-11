import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from rag.retrieval import retrieve

THRESHOLD = 0.7
JUDGE_MODEL = "gpt-4o-mini"        # DeepEval takes a model name string
TOP_K = 5
GOLDEN = Path("eval/retrieval_golden.jsonl")

golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
answerable = [g for g in golden if g["type"] != "unanswerable"]

test_cases = []
for item in answerable:
    results = retrieve(item["question"], top_k=TOP_K)
    test_cases.append(
        LLMTestCase(
            input=item["question"],
            actual_output=item["reference_answer"],          # placeholder: no generator yet
            expected_output="\n".join(item["key_facts"]),    # one fact per line → recall = facts covered
            retrieval_context=[r.text for r in results],     # rank order matters for precision
            additional_metadata={
                "id": item["id"],
                "type": item["type"],
                "retrieved": [r.citation for r in results],
            },
        )
    )

metrics = [
    ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]

result = evaluate(test_cases=test_cases, metrics=metrics)

# ---- summary
scores: dict[str, list[float]] = {}
for tr in result.test_results:
    for md in tr.metrics_data:
        scores.setdefault(md.name, []).append(md.score or 0.0)

print("\n=== Retrieval summary ===")
for name, vals in scores.items():
    passed = sum(v >= THRESHOLD for v in vals)
    print(f"{name:<28} mean={sum(vals)/len(vals):.3f}  passed={passed}/{len(vals)}")