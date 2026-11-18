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
  /** saved conversation: the server loads history itself and stores both turns */
  conversation_id?: string | null;
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
  /** messages fetched from the server (the list endpoint only returns titles) */
  loaded: boolean;
  /** not saved on the server (the database was unavailable): history is sent from the browser */
  local?: boolean;
};

/** GET /conversations */
export type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

/** A saved message, as GET /conversations/{id} returns it. */
export type SavedMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  meta: (Partial<AnswerMeta> & { stopped?: boolean }) | null;
  created_at: string;
};

export type ConversationDetail = ConversationSummary & { messages: SavedMessage[] };

export type User = {
  id: string;
  email: string;
  created_at: string;
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
  database: boolean;
};
