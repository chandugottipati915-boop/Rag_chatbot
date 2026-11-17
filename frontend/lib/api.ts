// Client for the FastAPI backend.
import { createSSEParser } from "./sse";
import type { AnswerMeta, ChatRequest, Health, Paper, Source } from "./types";

// NEXT_PUBLIC_ variables are inlined into the browser bundle at build time.
export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export class ApiError extends Error {}

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
    case 422:
      return "The question was rejected: it must be 1–2,000 characters.";
    case 502:
      return "The language model call failed. Check OPENAI_API_KEY in the backend's .env.";
    case 503:
      return "Search is unavailable. Check that Qdrant is running (docker ps).";
    default:
      return `The API returned ${res.status}${detail ? `: ${detail}` : ""}.`;
  }
}

function networkError(): ApiError {
  return new ApiError(
    `Can't reach the API at ${API_URL}. Start it with: uv run uvicorn rag.api.main:app --app-dir src --reload`,
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
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal,
    });
  } catch (e) {
    if (signal.aborted) throw e;
    throw networkError();
  }
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
  const res = await fetch(`${API_URL}/papers`);
  if (!res.ok) throw new ApiError(await describeFailure(res));
  return res.json();
}

export async function getHealth(): Promise<Health | null> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
