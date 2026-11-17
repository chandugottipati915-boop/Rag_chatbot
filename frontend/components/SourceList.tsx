"use client";

import { useEffect, useRef } from "react";
import type { Source } from "@/lib/types";

function pages(s: Source) {
  return s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`;
}

/**
 * The passages an answer was built from. The active one (whose citation was
 * clicked) opens up and gets the highlighter treatment.
 */
export default function SourceList({
  sources,
  activeN,
  onSelect,
  idPrefix,
}: {
  sources: Source[];
  activeN: number | null;
  onSelect: (n: number | null) => void;
  idPrefix: string;
}) {
  const listRef = useRef<HTMLOListElement>(null);

  // bring the clicked source into view
  useEffect(() => {
    if (activeN == null) return;
    const el = document.getElementById(`${idPrefix}-${activeN}`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [activeN, idPrefix]);

  return (
    <ol ref={listRef} className="space-y-3">
      {sources.map((s) => {
        const active = s.n === activeN;
        return (
          <li
            key={s.n}
            id={`${idPrefix}-${s.n}`}
            className={`rounded-md border bg-sheet transition-colors ${
              active ? "border-ink/40" : "border-rule"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelect(active ? null : s.n)}
              aria-expanded={active}
              className="flex w-full items-start gap-3 px-3 py-2.5 text-left"
            >
              <span
                className={`mt-0.5 flex h-5 min-w-5 items-center justify-center rounded-sm px-1 text-xs font-semibold tabular-nums ${
                  active ? "bg-mark text-ink" : "bg-paper text-muted"
                }`}
              >
                {s.n}
              </span>
              <span className="min-w-0">
                <span className="block text-sm leading-snug font-medium text-ink">{s.title}</span>
                <span className="mt-0.5 block text-xs text-muted">
                  {s.section}, {pages(s)}
                </span>
              </span>
            </button>

            {active && (
              <div className="border-t border-rule px-3 pt-2.5 pb-3">
                {s.text ? (
                  <p className="passage-active font-serif text-[0.92rem] leading-relaxed text-ink">{s.text}</p>
                ) : (
                  <p className="text-sm text-muted">The passage text wasn&apos;t sent with this answer.</p>
                )}
                <a
                  href={`https://arxiv.org/abs/${s.arxiv_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-2.5 inline-block text-xs font-medium text-teal underline underline-offset-2 hover:text-teal-dark"
                >
                  Open the paper on arXiv
                </a>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
