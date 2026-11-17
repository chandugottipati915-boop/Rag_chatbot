import sys, pathlib  # make "src/" importable even if the editable install is skipped (macOS hidden .pth)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

"""Multi-turn eval: does the graph handle follow-up questions?

Each conversation in goldens/multiturn_golden.jsonl is run through the real graph twice:
  with_history  the earlier turns are sent, so the rewrite node can resolve "it", "the two"...
  no_history    only the follow-up is sent: what would happen without rewriting

Scores
  direct (no LLM): paper_hit, paper_recall, filter_ok, refused_ok
  LLM judges:      Rewrite Match (with_history only), Faithfulness, Correct Behavior

Usage
  uv run python eval/eval_multiturn.py              # all conversations, both runs
  uv run python eval/eval_multiturn.py --limit 3    # quick check while developing
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from rag.config import PROJECT_ROOT
from rag.generation import NO_ANSWER
from rag.graph.build import get_graph

THRESHOLD = 0.7
JUDGE_MODEL = "gpt-4o-mini"
GOLDEN = PROJECT_ROOT / "goldens" / "multiturn_golden.jsonl"
OUT_DIR = PROJECT_ROOT / "eval" / "results"
RUNS = ("with_history", "no_history")

# Same behavior descriptions as eval/eval_generator.py, so "Correct Behavior" means the same thing
BEHAVIOR_SPEC = {
    "answer": "ANSWER. The papers contain the answer. The response should answer the question. "
              "Refusing ('I don't know') is wrong.",
    "correct_premise": "CORRECT THE PREMISE. The question contains a false assumption. The response must "
                       "say the assumption is wrong and give the correct information. Refusing, or "
                       "accepting the false assumption, is wrong.",
    "abstain": "DECLINE. The papers do not contain the answer. The response must decline (for example "
               "'I don't know based on the provided papers') and must not answer from general knowledge "
               "or invent facts.",
}


# ------------------------------------------------------------------ 1. run the graph

def run_case(case: dict, with_history: bool) -> dict:
    inp = {"question": case["question"]}
    if with_history:
        inp["history"] = case["history"]
    state = get_graph().invoke(inp)
    chunks = state.get("chunks", [])
    return {
        "id": case["id"], "type": case["type"],
        "run": "with_history" if with_history else "no_history",
        "expected_behavior": case["expected_behavior"],
        "question": case["question"],
        "expected_query": case["expected_query"],
        "query": state.get("query"),
        "paper_filter": state.get("paper_filter"),
        "expected_paper_filter": case["expected_paper_filter"],
        "source_papers": sorted({c.arxiv_id for c in chunks}),
        "expected_papers": case["expected_papers"],
        "chunk_texts": [c.text for c in chunks],
        "answer": state["answer"],
        "retries": state.get("retries", 0),
    }


# ------------------------------------------------------------------ 2. direct checks (no LLM)

def direct_checks(r: dict) -> None:
    expected, found = set(r["expected_papers"]), set(r["source_papers"])
    refused = r["answer"].strip() == NO_ANSWER
    if expected:                                       # answer / correct_premise cases
        r["paper_hit"] = float(bool(expected & found))
        r["paper_recall"] = len(expected & found) / len(expected)
    # refusing is right only when the papers can't answer
    r["refused_ok"] = float(refused == (r["expected_behavior"] == "abstain"))
    if r["run"] == "with_history":                     # without history there's nothing to resolve
        # no filter is harmless (retrieval searches every paper); a WRONG filter hides the answer
        r["filter_ok"] = float(r["paper_filter"] in (None, r["expected_paper_filter"]))


# ------------------------------------------------------------------ 3. LLM judges

def history_text(case: dict) -> str:
    turns = [f"{m['role'].capitalize()}: {m['content']}" for m in case["history"]]
    return "\n".join(turns + [f"User (latest question): {case['question']}"])


rewrite_metric = GEval(
    name="Rewrite Match",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                       SingleTurnParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "The input is a conversation; the actual output is a search query written for its latest "
        "question; the expected output is a correct standalone version of that question.",
        "Check that the actual output names the same subject (paper, method or object) as the expected "
        "output. A wrong or missing subject gets a very low score.",
        "Check that it asks the same thing as the expected output. Different wording is fine.",
        "Lower the score if it still contains unresolved references ('it', 'they', 'the two', "
        "'the first one') or carries over a subject from an earlier, different topic.",
    ],
    model=JUDGE_MODEL,
    threshold=THRESHOLD,
)

behavior_metric = GEval(
    name="Correct Behavior",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                       SingleTurnParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Read the 'Expected behavior' line in the expected output: ANSWER, CORRECT THE PREMISE or DECLINE.",
        "Decide which of these behaviors the actual output shows.",
        "If the actual output shows a different behavior than expected, give a very low score.",
        "If the behavior matches and is ANSWER or CORRECT THE PREMISE, check the content against the "
        "key facts: lower the score for contradictions, wrong facts or missing key points.",
        "If the behavior matches and is DECLINE, do not penalize the response for being short.",
    ],
    model=JUDGE_MODEL,
    threshold=THRESHOLD,
)

faithfulness_metric = FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True)


def case_name(r: dict) -> str:
    return f"{r['id']}-{r['run']}"     # unique: the same question appears in both runs


def build_test_cases(results: list[dict], cases: dict) -> tuple[list, list, list]:
    rewrite_tcs, answer_tcs, refusal_tcs = [], [], []
    for r in results:
        case = cases[r["id"]]
        if r["run"] == "with_history":
            rewrite_tcs.append(LLMTestCase(
                name=case_name(r), input=history_text(case),
                actual_output=r["query"] or "", expected_output=case["expected_query"],
            ))
        facts = "\n".join(f"- {f}" for f in case["key_facts"]) or "(none)"
        tc = LLMTestCase(
            name=case_name(r),
            input=history_text(case) if r["run"] == "with_history" else case["question"],
            actual_output=r["answer"],
            expected_output=(f"Expected behavior: {BEHAVIOR_SPEC[case['expected_behavior']]}\n"
                             f"Key facts:\n{facts}"),
            retrieval_context=r["chunk_texts"] or None,
        )
        # Faithfulness needs retrieved context; a refusal with no chunks gets Correct Behavior only
        (answer_tcs if r["chunk_texts"] else refusal_tcs).append(tc)
    return rewrite_tcs, answer_tcs, refusal_tcs


def judge(test_cases: list, metrics: list) -> dict[str, dict[str, float]]:
    """Run DeepEval; return {case name: {metric name: score}}."""
    if not test_cases:
        return {}
    out = defaultdict(dict)
    for tr in evaluate(test_cases=test_cases, metrics=metrics).test_results:
        for md in tr.metrics_data or []:
            out[tr.name][md.name.replace(" [GEval]", "")] = md.score or 0.0
    return out


# ------------------------------------------------------------------ 4. summary

def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):.2f}" if values else "  - "


# result key -> short column label
COLUMNS = {"paper_hit": "paper hit", "paper_recall": "recall", "filter_ok": "filter ok",
           "refused_ok": "refusal ok", "Rewrite Match": "rewrite", "Faithfulness": "faithful",
           "Correct Behavior": "behavior"}


def summarize(results: list[dict]) -> None:
    cols = list(COLUMNS)
    header = "".join(f"{COLUMNS[c]:>11}" for c in cols)
    print(f"\n{'=' * 94}\nMULTI-TURN EVAL  judge={JUDGE_MODEL}\n{'=' * 94}")
    print(f"{'run':<14}{'n':>3}" + header)
    for run in RUNS:
        rows = [r for r in results if r["run"] == run]
        if rows:
            print(f"{run:<14}{len(rows):>3}" + "".join(
                f"{mean([r[c] for r in rows if c in r]):>11}" for c in cols))

    print("\nWith history, by type:")
    print(f"{'type':<14}{'n':>3}" + header)
    for t in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["run"] == "with_history" and r["type"] == t]
        if rows:
            print(f"{t:<14}{len(rows):>3}" + "".join(
                f"{mean([r[c] for r in rows if c in r]):>11}" for c in cols))

    print("\nWith-history failures (wrong paper, wrong filter, wrong refusal, or a weak rewrite):")
    for r in results:
        if r["run"] != "with_history":
            continue
        problems = [k for k in ("paper_hit", "filter_ok", "refused_ok") if r.get(k) == 0.0]
        if r.get("Rewrite Match", 1.0) < THRESHOLD:
            problems.append("rewrite")
        if problems:
            print(f"  {r['id']} [{r['type']}] {', '.join(problems)}"
                  f"\n      query:  {r['query']!r}  (filter={r['paper_filter']}, expected {r['expected_paper_filter']})"
                  f"\n      papers: {r['source_papers']}  (expected {r['expected_papers']})")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N conversations")
    args = ap.parse_args()

    cases = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]
    if args.limit:
        cases = cases[: args.limit]
    by_id = {c["id"]: c for c in cases}

    results = []
    for i, case in enumerate(cases, 1):
        for with_history in (True, False):
            run = "with_history" if with_history else "no_history"
            print(f"[{i}/{len(cases)}] {case['id']} {run:<12} {case['question'][:55]}")
            r = run_case(case, with_history)
            direct_checks(r)
            results.append(r)

    rewrite_tcs, answer_tcs, refusal_tcs = build_test_cases(results, by_id)
    scores = {}
    for part in (judge(rewrite_tcs, [rewrite_metric]),
                 judge(answer_tcs, [faithfulness_metric, behavior_metric]),
                 judge(refusal_tcs, [behavior_metric])):
        for name, metric_scores in part.items():
            scores.setdefault(name, {}).update(metric_scores)
    for r in results:
        r.update(scores.get(case_name(r), {}))

    summarize(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"multiturn_{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    out.write_text("\n".join(json.dumps({k: v for k, v in r.items() if k != "chunk_texts"})
                             for r in results) + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
