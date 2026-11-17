"use client";

import Answer from "./Answer";
import SourceList from "./SourceList";
import type { ChatMessage } from "@/lib/types";

const REFUSAL = "I don't know based on the provided papers.";

export function UserMessage({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] rounded-lg bg-paper px-4 py-2.5 text-[0.95rem] leading-relaxed whitespace-pre-wrap text-ink">
        {message.content}
      </p>
    </div>
  );
}

export function AssistantMessage({
  message,
  activeN,
  focused,
  onCite,
  onFocus,
}: {
  message: ChatMessage;
  activeN: number | null;
  focused: boolean;
  onCite: (n: number | null) => void;
  onFocus: () => void;
}) {
  const sources = message.sources ?? [];
  const streaming = message.status === "streaming";
  const refused = message.content.trim() === REFUSAL;

  if (streaming && !message.content) {
    return (
      <p className="text-sm text-muted" role="status">
        {sources.length ? `Found ${sources.length} passages. Writing the answer…` : "Searching the papers…"}
      </p>
    );
  }

  return (
    <article className="space-y-3">
      {refused ? (
        <div className="rounded-md border border-dashed border-rule px-4 py-3">
          <p className="font-serif text-ink">The papers don&apos;t answer this.</p>
          <p className="mt-1 text-sm text-muted">
            Nothing relevant came back from the papers. Try naming a paper, or rephrase with the terms a paper would use.
          </p>
        </div>
      ) : (
        message.content && (
          <Answer text={message.content} sources={sources} activeN={activeN} onCite={onCite} streaming={streaming} />
        )
      )}

      {message.status === "stopped" && <p className="text-sm text-muted">Stopped.</p>}
      {message.status === "error" && (
        <p role="alert" className="rounded-md bg-danger/8 px-3 py-2 text-sm text-danger">
          {message.error}
        </p>
      )}

      {message.status === "done" && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
          {message.meta && <span>Answered in {(message.meta.latency_ms / 1000).toFixed(1)} s</span>}
          {message.meta?.rewritten_query && (
            <span title="Follow-up questions are rewritten into a standalone search">
              Searched for &ldquo;{message.meta.rewritten_query}&rdquo;
            </span>
          )}
          {!!message.meta?.retries && <span>Reworded the search once</span>}
          {!refused && sources.length > 0 && (
            <button
              type="button"
              onClick={onFocus}
              className={`hidden font-medium underline-offset-2 hover:underline lg:inline ${focused ? "text-ink" : "text-teal"}`}
            >
              {focused ? `${sources.length} sources in the margin` : `Show ${sources.length} sources`}
            </button>
          )}
        </div>
      )}

      {/* small screens: no margin column, so sources sit under the answer */}
      {!refused && sources.length > 0 && !streaming && (
        <details className="lg:hidden" open={activeN != null}>
          <summary className="cursor-pointer text-sm font-medium text-teal">
            {sources.length} sources
          </summary>
          <div className="mt-3">
            <SourceList sources={sources} activeN={activeN} onSelect={onCite} idPrefix={`inline-${message.id}`} />
          </div>
        </details>
      )}
    </article>
  );
}
