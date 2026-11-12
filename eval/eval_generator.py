import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import json
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from rag.config import PROJECT_ROOT
from rag.generation import generate
from rag.retrieval import retrieve

THRESHOLD = 0.7
JUDGE_MODEL = "gpt-4o-mini"
TOP_K = 5
GOLDEN = PROJECT_ROOT / "goldens" / "generator_golden.jsonl"
OUT_DIR = PROJECT_ROOT / "eval" / "results"

golden = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]

# ---- 1. run the real pipeline: retrieve → generate
test_cases, records = [], []
for i, item in enumerate(golden, 1):
    print(f"[{i}/{len(golden)}] {item['id']} {item['question'][:60]}")
    chunks = retrieve(item["question"], top_k=TOP_K)
    answer = generate(item["question"], chunks)

    test_cases.append(
        LLMTestCase(
            input=item["question"],
            actual_output=answer,                               # what the generator actually said
            expected_output=item["reference_answer"],           # not used by these metrics; shown in reports
            retrieval_context=[c.text for c in chunks],         # what Faithfulness checks claims against
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
    })

# save the generated answers: they cost money, and you'll want to read them
OUT_DIR.mkdir(parents=True, exist_ok=True)
answers_file = OUT_DIR / f"generator_answers_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
answers_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
print(f"\nSaved answers → {answers_file}")

# ---- 2. judge them
metrics = [
    FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
]
result = evaluate(test_cases=test_cases, metrics=metrics)

# ---- 3. summary, grouped by expected behavior
scores = defaultdict(lambda: defaultdict(list))   # behavior → metric → [scores]
for tc, tr in zip(test_cases, result.test_results):
    behavior = tc.additional_metadata["expected_behavior"]
    for md in tr.metrics_data:
        scores[behavior][md.name].append(md.score or 0.0)

print("\n=== Generator eval (judge: %s, k=%d) ===" % (JUDGE_MODEL, TOP_K))
for behavior in ("answer", "correct_premise", "abstain", "clarify_or_cover"):
    if behavior not in scores:
        continue
    print(f"\n{behavior}  (n={len(next(iter(scores[behavior].values())))})")
    for name, vals in scores[behavior].items():
        passed = sum(v >= THRESHOLD for v in vals)
        print(f"  {name:<20} mean={sum(vals)/len(vals):.3f}  passed={passed}/{len(vals)}")