"use client";

import { API_LABEL, API_URL } from "@/lib/api";
import type { Health, Paper } from "@/lib/types";

const EXAMPLES = [
  "How do RAG and Self-RAG differ in deciding when to retrieve?",
  "Why doesn't adding more retrieved documents always help, according to Lost in the Middle?",
  "Why doesn't HyDE need any relevance labels?",
  "What is the difference between faithfulness and answer relevance in RAGAS?",
];

export default function EmptyState({
  health,
  papers,
  onPick,
}: {
  health: Health | null | undefined;
  papers: Paper[];
  onPick: (q: string) => void;
}) {
  return (
    <div className="mx-auto w-full max-w-[42rem] px-5 pt-[12vh] pb-10">
      <h1 className="font-serif text-[2.4rem] leading-[1.1] font-semibold tracking-tight text-ink sm:text-[2.9rem]">
        Ask the papers.
      </h1>
      <p className="mt-3 max-w-[34rem] text-[1.02rem] leading-relaxed text-muted">
        Answers come from {papers.length || 15} papers on retrieval-augmented generation, with a numbered citation for
        every claim. Click a number to read the passage it came from.
      </p>

      {health === null && (
        <div role="alert" className="mt-8 rounded-md border border-danger/30 bg-sheet px-4 py-3.5 text-sm">
          <p className="font-medium text-danger">Can&apos;t reach the API at {API_LABEL}.</p>
          {API_URL ? (
            <>
              <p className="mt-1.5 text-muted">Start Qdrant and the backend from the project folder:</p>
              <pre className="mt-2 overflow-x-auto rounded bg-paper px-3 py-2 text-xs leading-relaxed text-ink">
                {`docker start qdrant\nuv run uvicorn rag.api.main:app --app-dir src --reload`}
              </pre>
            </>
          ) : (
            <p className="mt-1.5 text-muted">If nobody used it for a while, it is waking up. Reload in a minute.</p>
          )}
        </div>
      )}

      <div className="mt-10">
        <p className="text-sm font-medium text-ink">Try one of these</p>
        <ul className="mt-3 divide-y divide-rule border-y border-rule">
          {EXAMPLES.map((q) => (
            <li key={q}>
              <button
                type="button"
                onClick={() => onPick(q)}
                className="w-full py-3 text-left font-serif text-[1.02rem] text-ink hover:text-teal"
              >
                {q}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {papers.length > 0 && (
        <details className="mt-8 text-sm">
          <summary className="cursor-pointer font-medium text-teal">See all {papers.length} papers</summary>
          <ul className="mt-3 space-y-1.5 text-muted">
            {papers.map((p) => (
              <li key={p.arxiv_id}>
                <a href={p.url} target="_blank" rel="noreferrer" className="hover:text-ink hover:underline">
                  {p.title}
                </a>
                {p.published && <span className="text-muted/70"> ({p.published.slice(0, 4)})</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
