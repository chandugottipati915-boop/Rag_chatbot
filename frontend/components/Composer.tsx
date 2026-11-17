"use client";

import { useEffect, useRef, useState } from "react";
import { SendIcon, StopIcon } from "./icons";
import type { Paper } from "@/lib/types";

export default function Composer({
  busy,
  papers,
  paperId,
  onPaperChange,
  onSend,
  onStop,
  prefill,
}: {
  busy: boolean;
  papers: Paper[];
  paperId: string | null;
  onPaperChange: (id: string | null) => void;
  onSend: (question: string) => void;
  onStop: () => void;
  prefill: { text: string; nonce: number } | null;
}) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  // grow with the text, up to ~8 lines
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  // an example question was picked: put it in the box
  useEffect(() => {
    if (!prefill) return;
    setText(prefill.text); // eslint-disable-line react-hooks/set-state-in-effect -- syncing from a parent event
    ref.current?.focus();
  }, [prefill]);

  const submit = () => {
    const q = text.trim();
    if (!q || busy) return;
    onSend(q);
    setText("");
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      className="rounded-lg border border-rule bg-sheet shadow-[0_1px_0_var(--color-rule)] focus-within:border-teal"
    >
      <label htmlFor="question" className="sr-only">
        Your question
      </label>
      <textarea
        id="question"
        ref={ref}
        rows={1}
        value={text}
        maxLength={2000}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder="Ask about the papers"
        className="block w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[0.98rem] leading-relaxed text-ink placeholder:text-muted/70 focus:outline-none focus-visible:outline-none"
      />
      <div className="flex items-center justify-between gap-3 px-3 pb-3">
        <label className="flex min-w-0 flex-1 items-center gap-2 text-xs text-muted">
          <span className="hidden shrink-0 sm:inline">Search in</span>
          <select
            value={paperId ?? ""}
            onChange={(e) => onPaperChange(e.target.value || null)}
            className="w-full max-w-[16rem] min-w-0 truncate rounded border border-rule bg-sheet px-2 py-1 text-xs text-ink"
          >
            <option value="">All {papers.length || ""} papers</option>
            {papers.map((p) => (
              <option key={p.arxiv_id} value={p.arxiv_id}>
                {p.title}
              </option>
            ))}
          </select>
        </label>

        {busy ? (
          <button
            type="button"
            onClick={onStop}
            className="flex shrink-0 h-9 items-center gap-2 rounded-md border border-rule px-3 text-sm font-medium text-ink hover:bg-paper"
          >
            <StopIcon /> Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!text.trim()}
            className="flex shrink-0 h-9 items-center gap-2 rounded-md bg-teal px-3.5 text-sm font-medium text-white hover:bg-teal-dark disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendIcon /> Ask
          </button>
        )}
      </div>
    </form>
  );
}
