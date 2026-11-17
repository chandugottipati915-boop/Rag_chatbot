// Chat history lives in the browser's localStorage for now.
// (A server-side store comes with auth, later in Phase 4.)
import type { Conversation } from "./types";

const KEY = "rag-chat.conversations.v1";
const MAX_CONVERSATIONS = 50;

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return []; // storage disabled or corrupted: start fresh
  }
}

export function saveConversations(conversations: Conversation[]) {
  try {
    const recent = [...conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS);
    localStorage.setItem(KEY, JSON.stringify(recent));
  } catch {
    /* quota exceeded or storage disabled: keep working in memory */
  }
}
