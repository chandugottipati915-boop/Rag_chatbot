import { describe, expect, it } from "vitest";
import { fromSaved, fromSummary, titleFor } from "../lib/conversations";

describe("saved conversations", () => {
  it("turns a summary into an unloaded conversation", () => {
    const c = fromSummary({ id: "a", title: "RAG", created_at: "2026-11-17T10:00:00Z", updated_at: "2026-11-17T11:00:00Z" });
    expect(c).toMatchObject({ id: "a", title: "RAG", loaded: false, messages: [] });
    expect(c.updatedAt).toBe(Date.parse("2026-11-17T11:00:00Z"));
  });

  it("restores answer metadata and the stopped state", () => {
    const base = { id: "m", role: "assistant" as const, content: "x", sources: null, created_at: "" };
    expect(fromSaved({ ...base, meta: { latency_ms: 4200, rewritten_query: null, retries: 1 } })).toMatchObject({
      status: "done",
      sources: [],
      meta: { latency_ms: 4200, rewritten_query: null, retries: 1 },
    });
    expect(fromSaved({ ...base, meta: { stopped: true } })).toMatchObject({ status: "stopped", meta: undefined });
  });

  it("shortens long titles", () => {
    expect(titleFor("a".repeat(80))).toHaveLength(58);
    expect(titleFor("short")).toBe("short");
  });
});
