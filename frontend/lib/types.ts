// Shapes shared with the FastAPI backend (src/rag/api/schemas.py) and the UI.

/** One retrieved passage. `n` is the number the answer cites as [n]. */
export type Source = {
  n: number;
  arxiv_id: string;
  title: string;
  section: string;
  page_start: number;
  page_end: number;
  score: number;
  citation: string;
  text?: string | null;
};

/** Sent to the API as conversation history (max 20 turns). */
export type HistoryMessage = { role: "user" | "assistant"; content: string };

export type ChatRequest = {
  question: string;
  top_k?: number;
  arxiv_id?: string | null;
  history: HistoryMessage[];
};

/** The `done` event: what the graph did on the way to the answer. */
export type AnswerMeta = {
  latency_ms: number;
  rewritten_query: string | null;
  retries: number;
};

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  meta?: AnswerMeta;
  status?: MessageStatus;
  error?: string;
};

export type Conversation = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
};

export type Paper = {
  arxiv_id: string;
  title: string;
  published: string | null;
  url: string;
};

export type Health = {
  status: "ok" | "degraded";
  qdrant: boolean;
  collection: string;
  points: number | null;
};
