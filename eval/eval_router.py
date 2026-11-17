import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Router eval: is each message sent down the right path (papers / chat / off_topic)?

The router runs inside the rewrite node: one small LLM call for a first message, or the
rewrite call itself for a follow-up. This eval calls that node directly, so it needs an
OpenAI key but no Qdrant, and runs in about a minute.

Cases
  goldens/router_golden.jsonl   greetings, thanks, questions about the bot, mixed messages,
                                off-topic questions (some with history)
  every question in the other golden sets, expected "papers": a paper question sent to
  "chat" or "off_topic" would never be searched. The unanswerable generator questions
  may also go to "off_topic" (both end in a refusal).

The costly mistake is a paper question routed away (a false refusal), so it's reported
separately. Exits with status 1 below --min-accuracy (for CI later).

Usage
  uv run python eval/eval_router.py
  uv run python eval/eval_router.py --min-accuracy 0.95
"""

import argparse
import json
import time
from collections import Counter

from rag.config import PROJECT_ROOT
from rag.graph.nodes import rewrite

GOLDENS = PROJECT_ROOT / "goldens"
ROUTES = ["papers", "chat", "off_topic"]


def load_cases() -> list[dict]:
    cases = [json.loads(line) for line in open(GOLDENS / "router_golden.jsonl")]
    for line in open(GOLDENS / "generator_golden.jsonl"):
        g = json.loads(line)
        expected = ["papers", "off_topic"] if g["expected_behavior"] == "abstain" else ["papers"]
        cases.append({"id": g["id"], "question": g["question"], "history": [], "expected": expected})
    for line in open(GOLDENS / "retrieval_golden.jsonl"):
        g = json.loads(line)
        cases.append({"id": g["id"], "question": g["question"], "history": [], "expected": ["papers"]})
    for line in open(GOLDENS / "multiturn_golden.jsonl"):
        g = json.loads(line)
        expected = ["off_topic", "papers"] if g["type"] == "off_topic" else ["papers"]
        cases.append({"id": g["id"], "question": g["question"], "history": g["history"], "expected": expected})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    args = ap.parse_args()

    cases = load_cases()
    confusion: Counter = Counter()
    wrong, missed_papers, times = [], [], []
    for case in cases:
        t0 = time.perf_counter()
        got = rewrite({"question": case["question"], "history": case["history"]})["route"]
        times.append(time.perf_counter() - t0)
        primary = case["expected"][0]
        confusion[(primary, got)] += 1
        if got not in case["expected"]:
            wrong.append((case, got))
            if primary == "papers" and "off_topic" not in case["expected"]:
                missed_papers.append((case, got))
        print(f"{'ok ' if got in case['expected'] else 'BAD'} {case['id']:>5}  {got:<9}  {case['question'][:70]}")

    accuracy = 1 - len(wrong) / len(cases)
    print(f"\n{len(cases)} cases, accuracy {accuracy:.3f}")
    print(f"paper questions routed away (would be false refusals): {len(missed_papers)}")
    times.sort()
    print(f"routing latency p50 {times[len(times) // 2]:.2f} s, p95 {times[int(len(times) * 0.95)]:.2f} s")

    print("\nexpected →   " + "  ".join(f"{r:>9}" for r in ROUTES) + "   (got)")
    for exp in ROUTES:
        print(f"{exp:>10}   " + "  ".join(f"{confusion[(exp, got)]:>9}" for got in ROUTES))

    if wrong:
        print("\nMisrouted:")
        for case, got in wrong:
            hist = " (with history)" if case["history"] else ""
            print(f"  {case['id']}: expected {'/'.join(case['expected'])}, got {got}{hist}: {case['question']}")

    sys.exit(0 if accuracy >= args.min_accuracy else 1)


if __name__ == "__main__":
    main()
