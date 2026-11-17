"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { linkCitations } from "@/lib/citations";
import type { Source } from "@/lib/types";

/** Markdown answer whose [n] citations become clickable highlighter marks. */
export default function Answer({
  text,
  sources,
  activeN,
  onCite,
  streaming,
}: {
  text: string;
  sources: Source[];
  activeN: number | null;
  onCite: (n: number) => void;
  streaming: boolean;
}) {
  const known = new Set(sources.map((s) => s.n));

  return (
    <div className={`answer ${streaming ? "caret" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a({ href, children }) {
            const match = href?.match(/^#cite-(\d+)$/);
            if (match) {
              const n = Number(match[1]);
              if (!known.has(n)) return <span className="text-muted">[{n}]</span>;
              const s = sources.find((x) => x.n === n)!;
              return (
                <button
                  type="button"
                  onClick={() => onCite(n)}
                  title={`${s.title}, ${s.section}`}
                  aria-label={`Show source ${n}: ${s.title}`}
                  className={`mx-px inline-flex h-[1.15em] min-w-[1.15em] -translate-y-px items-center justify-center rounded-[3px] px-[0.25em] align-baseline font-sans text-[0.72em] font-semibold tabular-nums text-ink transition-colors ${
                    n === activeN ? "bg-mark" : "bg-mark-soft hover:bg-mark"
                  }`}
                >
                  {n}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="text-teal underline underline-offset-2">
                {children}
              </a>
            );
          },
        }}
      >
        {linkCitations(text)}
      </ReactMarkdown>
    </div>
  );
}
