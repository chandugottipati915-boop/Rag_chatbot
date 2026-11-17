// Convert between the API's saved shapes and the UI's.
import type { ChatMessage, Conversation, ConversationSummary, SavedMessage } from "./types";

export function fromSummary(c: ConversationSummary): Conversation {
  return {
    id: c.id,
    title: c.title,
    createdAt: Date.parse(c.created_at),
    updatedAt: Date.parse(c.updated_at),
    messages: [],
    loaded: false,
  };
}

export function fromSaved(m: SavedMessage): ChatMessage {
  const meta = m.meta ?? {};
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    sources: m.sources ?? [],
    status: meta.stopped ? "stopped" : "done",
    meta:
      meta.latency_ms != null
        ? { latency_ms: meta.latency_ms, rewritten_query: meta.rewritten_query ?? null, retries: meta.retries ?? 0 }
        : undefined,
  };
}

export function titleFor(question: string) {
  return question.length > 60 ? `${question.slice(0, 57)}…` : question;
}
