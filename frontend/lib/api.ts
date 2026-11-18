// Client for the FastAPI backend.
import { createSSEParser } from "./sse";
import type {
  AnswerMeta, ChatRequest, ConversationDetail, ConversationSummary, Health, Paper, Source, User,
} from "./types";

// Where the API is. In production the API serves this app itself, so requests go to the
// same origin (""). In development the API runs separately on port 8000.
// NEXT_PUBLIC_API_URL overrides both; NEXT_PUBLIC_ variables are inlined at build time.
const DEFAULT_API_URL = process.env.NODE_ENV === "production" ? "" : "http://localhost:8000";
export const API_URL = (process.env.NEXT_PUBLIC_API_URL || DEFAULT_API_URL).replace(/\/$/, "");

/** How to name the API in messages: "" (same origin) reads badly. */
export const API_LABEL = API_URL || "this server";

export class ApiError extends Error {}

export const QUESTION_REJECTED = "The question was rejected: it must be 1–2,000 characters.";

/** 401: not signed in, or the session expired. The UI goes back to the sign-in screen. */
export class AuthError extends ApiError {}

// The session is an httpOnly cookie set by the API. "include" makes the browser send it
// to the API (a different origin: port 8000) and accept the cookies the API sets.
const CREDENTIALS: RequestCredentials = "include";

export type StreamHandlers = {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (meta: AnswerMeta) => void;
};

/** Turn an HTTP failure into a message that says what to do about it. */
async function describeFailure(res: Response): Promise<string> {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : "";
  } catch {
    /* body wasn't JSON */
  }
  switch (res.status) {
    case 401:
      return detail || "Sign in to continue.";
    case 409:
      return detail || "That already exists.";
    case 429:
      return detail || "Too many attempts. Try again in a few minutes.";
    case 404:
      return "This chat no longer exists on the server. Start a new chat.";
    case 422:
      return QUESTION_REJECTED;
    case 502:
      return "The language model call failed. Check OPENAI_API_KEY in the backend's .env.";
    case 503:
      return API_URL
        ? "Search is unavailable. Check that Qdrant is running (docker ps)."
        : "Search is unavailable right now: the vector database can't be reached or has no papers loaded.";
    default:
      return `The API returned ${res.status}${detail ? `: ${detail}` : ""}.`;
  }
}

function networkError(): ApiError {
  return new ApiError(
    API_URL
      ? `Can't reach the API at ${API_URL}. Start it with: uv run uvicorn rag.api.main:app --app-dir src --reload`
      : "Can't reach the server. It may be waking up: free hosting sleeps when idle. Try again in a minute.",
  );
}

/**
 * POST /chat/stream and dispatch its events in order:
 * sources → token… → done   (or an `error` event if the LLM fails mid-stream).
 */
export async function streamChat(req: ChatRequest, handlers: StreamHandlers, signal: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      credentials: CREDENTIALS,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    if (signal.aborted) throw e;
    throw networkError();
  }
  if (res.status === 401) throw new AuthError(await describeFailure(res));
  if (!res.ok || !res.body) throw new ApiError(await describeFailure(res));

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parse = createSSEParser();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const { event, data } of parse(value)) {
      const payload = JSON.parse(data);
      if (event === "sources") handlers.onSources(payload as Source[]);
      else if (event === "token") handlers.onToken(payload as string);
      else if (event === "done") handlers.onDone(payload as AnswerMeta);
      else if (event === "error") throw new ApiError("The language model stopped mid-answer. Try again.");
    }
  }
}

export async function getPapers(): Promise<Paper[]> {
  const res = await fetch(`${API_URL}/papers`, { credentials: CREDENTIALS });
  if (!res.ok) throw new ApiError(await describeFailure(res));
  return res.json();
}

export async function getHealth(): Promise<Health | null> {
  try {
    const res = await fetch(`${API_URL}/health`, { credentials: CREDENTIALS });
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- saved conversations

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      credentials: CREDENTIALS,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw networkError();
  }
  if (res.status === 401) throw new AuthError(await describeFailure(res));
  if (!res.ok) throw new ApiError(await describeFailure(res));
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const listConversations = () => request<ConversationSummary[]>("/conversations");

export const getConversation = (id: string) => request<ConversationDetail>(`/conversations/${id}`);

export const createConversation = (title: string) =>
  request<ConversationSummary>("/conversations", { method: "POST", body: JSON.stringify({ title }) });

export const deleteConversation = (id: string) =>
  request<void>(`/conversations/${id}`, { method: "DELETE" });

// ---------------------------------------------------------------- auth

export const signUp = (email: string, password: string) =>
  request<User>("/auth/signup", { method: "POST", body: JSON.stringify({ email, password }) });

export const signIn = (email: string, password: string) =>
  request<User>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });

export const signOut = () => request<void>("/auth/logout", { method: "POST" });

/** The signed-in user, or null when nobody is (throws only if the API is unreachable). */
export async function getMe(): Promise<User | null> {
  try {
    return await request<User>("/auth/me");
  } catch (e) {
    if (e instanceof AuthError) return null;
    throw e;
  }
}
