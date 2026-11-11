"""Retrieval evaluation with an LLM as judge.

Metrics per question (answerable questions):
  paper_hit        any retrieved chunk comes from a relevant paper        (deterministic)
  paper_recall     share of relevant papers that appear in the top-k     (deterministic)
  precision@k      share of retrieved chunks the judge marks relevant    (LLM judge)
  avg_precision    rank-aware precision: rewards relevant chunks ranked high (LLM judge)
  mrr              1 / rank of the first relevant chunk                  (LLM judge)
  context_recall   share of key facts supported by the retrieved chunks  (LLM judge)

Unanswerable questions:
  false_positive_rate  share of retrieved chunks the judge marks relevant (should be ~0)
  top_score            similarity of the best chunk (helps pick a min_score cutoff)

Usage:
  uv run python eval/run_retrieval_eval.py                 # all questions, k=5
  uv run python eval/run_retrieval_eval.py --k 10 --tag k10
  uv run python eval/run_retrieval_eval.py --limit 3       # quick smoke test

Judge settings (.env), any OpenAI-compatible endpoint works:
  JUDGE_MODEL=gpt-4o-mini
  JUDGE_API_KEY=sk-...             # falls back to OPENAI_API_KEY
  JUDGE_BASE_URL=                  # e.g. http://localhost:11434/v1 for Ollama
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.retrieval import RetrievedChunk, retrieve

load_dotenv()

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "retrieval_golden.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_FILE = RESULTS_DIR / "judge_cache.json"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
judge = OpenAI(
    api_key=os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed",
    base_url=os.getenv("JUDGE_BASE_URL") or None,
)

# --------------------------------------------------------------------------- prompts

RELEVANCE_PROMPT = """You are evaluating a search system over research papers.

Question: {question}
Reference answer: {reference}

Retrieved passage:
\"\"\"
{chunk}
\"\"\"

Is this passage RELEVANT? A passage is relevant only if it contains information that
directly helps answer the question (fully or partially). A passage that merely mentions
the same topic or paper, without information needed for the answer, is NOT relevant.

Respond with JSON only: {{"relevant": true or false, "reason": "<one short sentence>"}}"""

RECALL_PROMPT = """You are evaluating a search system over research papers.

Question: {question}

Key facts a complete answer needs:
{facts}

Retrieved passages:
{passages}

For EACH key fact, decide whether it is supported by (can be inferred from) the retrieved
passages. Do not use outside knowledge: only what the passages say.

Respond with JSON only:
{{"facts": [{{"fact": <fact number>, "supported": true or false, "passages": [<passage numbers>]}}]}}"""

# --------------------------------------------------------------------------- judge helpers


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


CACHE = _load_cache()


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def ask_judge(prompt: str, retries: int = 3) -> dict:
    key = hashlib.sha256(f"{JUDGE_MODEL}\n{prompt}".encode()).hexdigest()
    if key in CACHE:
        return CACHE[key]
    for attempt in range(retries):
        try:
            resp = judge.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            result = _parse_json(resp.choices[0].message.content)
            CACHE[key] = result
            return result
        except Exception as e:  # network or JSON errors
            if attempt == retries - 1:
                raise
            print(f"    judge retry ({e.__class__.__name__}) ...")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def judge_relevance(q: dict, chunk: RetrievedChunk) -> bool:
    out = ask_judge(RELEVANCE_PROMPT.format(
        question=q["question"], reference=q["reference_answer"], chunk=chunk.text,
    ))
    return bool(out.get("relevant"))


def judge_recall(q: dict, chunks: list[RetrievedChunk]) -> list[bool]:
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(q["key_facts"], 1))
    passages = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
    out = ask_judge(RECALL_PROMPT.format(question=q["question"], facts=facts, passages=passages))
    supported = {int(f["fact"]): bool(f.get("supported")) for f in out.get("facts", [])}
    return [supported.get(i, False) for i in range(1, len(q["key_facts"]) + 1)]

# --------------------------------------------------------------------------- metrics


def average_precision(rels: list[bool]) -> float:
    """Mean of precision@i over the ranks i where a relevant chunk appears."""
    hits, total = 0, 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def reciprocal_rank(rels: list[bool]) -> float:
    return next((1 / i for i, r in enumerate(rels, 1) if r), 0.0)


def evaluate_question(q: dict, k: int) -> dict:
    chunks = retrieve(q["question"], top_k=k)
    rels = [judge_relevance(q, c) for c in chunks]
    retrieved_papers = [c.arxiv_id for c in chunks]

    row = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "retrieved": [
            {"rank": i, "arxiv_id": c.arxiv_id, "section": c.section,
             "page": c.page_start, "score": round(c.score, 4), "relevant": r}
            for i, (c, r) in enumerate(zip(chunks, rels), 1)
        ],
        "top_score": round(chunks[0].score, 4) if chunks else 0.0,
    }

    if q["type"] == "unanswerable":
        row["false_positive_rate"] = sum(rels) / len(rels) if rels else 0.0
        return row

    relevant_papers = set(q["relevant_papers"])
    fact_support = judge_recall(q, chunks) if chunks else [False] * len(q["key_facts"])
    row.update({
        "paper_hit": float(bool(relevant_papers & set(retrieved_papers))),
        "paper_recall": len(relevant_papers & set(retrieved_papers)) / len(relevant_papers),
        "precision_at_k": sum(rels) / len(rels) if rels else 0.0,
        "avg_precision": average_precision(rels),
        "mrr": reciprocal_rank(rels),
        "context_recall": sum(fact_support) / len(fact_support) if fact_support else 0.0,
        "fact_support": fact_support,
    })
    return row


ANSWERABLE_METRICS = ["paper_hit", "paper_recall", "precision_at_k", "avg_precision", "mrr", "context_recall"]


def summarize(rows: list[dict]) -> dict:
    summary = {}
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    for group, subset in [("all_answerable", answerable),
                          ("single_paper", [r for r in answerable if r["type"] == "single_paper"]),
                          ("multi_paper", [r for r in answerable if r["type"] == "multi_paper"])]:
        if subset:
            summary[group] = {m: round(statistics.mean(r[m] for r in subset), 3) for m in ANSWERABLE_METRICS}
            summary[group]["n"] = len(subset)
    unans = [r for r in rows if r["type"] == "unanswerable"]
    if unans:
        summary["unanswerable"] = {
            "false_positive_rate": round(statistics.mean(r["false_positive_rate"] for r in unans), 3),
            "mean_top_score": round(statistics.mean(r["top_score"] for r in unans), 3),
            "n": len(unans),
        }
    if answerable:
        summary["answerable_mean_top_score"] = round(statistics.mean(r["top_score"] for r in answerable), 3)
    return summary


def print_report(summary: dict, rows: list[dict], k: int) -> None:
    print(f"\n{'=' * 72}\nRETRIEVAL EVAL  k={k}  judge={JUDGE_MODEL}\n{'=' * 72}")
    header = f"{'group':<16}{'n':>4}" + "".join(f"{m:>15}" for m in ANSWERABLE_METRICS)
    print(header)
    for group in ("all_answerable", "single_paper", "multi_paper"):
        if group in summary:
            s = summary[group]
            print(f"{group:<16}{s['n']:>4}" + "".join(f"{s[m]:>15.3f}" for m in ANSWERABLE_METRICS))
    if "unanswerable" in summary:
        u = summary["unanswerable"]
        print(f"\nunanswerable (n={u['n']}): false_positive_rate={u['false_positive_rate']:.3f}  "
              f"mean_top_score={u['mean_top_score']:.3f}  "
              f"(answerable mean_top_score={summary.get('answerable_mean_top_score', 0):.3f})")

    worst = sorted((r for r in rows if r["type"] != "unanswerable"),
                   key=lambda r: (r["context_recall"], r["precision_at_k"]))[:5]
    print("\nWeakest questions:")
    for r in worst:
        top = r["retrieved"][0] if r["retrieved"] else {}
        print(f"  {r['id']}  recall={r['context_recall']:.2f}  p@k={r['precision_at_k']:.2f}  "
              f"top1={top.get('arxiv_id')} {top.get('section', '')[:30]!r}  | {r['question'][:55]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--tag", default="baseline", help="label for this run, e.g. hybrid, rerank")
    args = ap.parse_args()

    questions = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} {q['question'][:70]}")
        rows.append(evaluate_question(q, args.k))
        CACHE_FILE.write_text(json.dumps(CACHE))          # save progress as we go

    summary = summarize(rows)
    print_report(summary, rows, args.k)

    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{args.tag}_k{args.k}.json"
    out.write_text(json.dumps(
        {"tag": args.tag, "k": args.k, "judge_model": JUDGE_MODEL,
         "summary": summary, "questions": rows}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
