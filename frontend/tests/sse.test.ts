import { describe, expect, it } from "vitest";
import { createSSEParser } from "../lib/sse";

describe("createSSEParser", () => {
  it("parses complete events", () => {
    const push = createSSEParser();
    expect(push('event: token\ndata: "Hi"\n\nevent: done\ndata: {"latency_ms": 5}\n\n')).toEqual([
      { event: "token", data: '"Hi"' },
      { event: "done", data: '{"latency_ms": 5}' },
    ]);
  });

  it("waits for the rest of an event split across chunks", () => {
    const push = createSSEParser();
    expect(push("event: tok")).toEqual([]);
    expect(push('en\ndata: "a b"\n')).toEqual([]);
    expect(push("\n")).toEqual([{ event: "token", data: '"a b"' }]);
  });

  it("handles CRLF line endings, even when split", () => {
    const push = createSSEParser();
    expect(push('event: token\r\ndata: "x"\r\n\r')).toEqual([]);
    expect(push("\n")).toEqual([{ event: "token", data: '"x"' }]);
  });

  it("ignores comments and keeps JSON-escaped newlines intact", () => {
    const push = createSSEParser();
    expect(push(': keep-alive\n\nevent: token\ndata: "line1\\nline2"\n\n')).toEqual([
      { event: "token", data: '"line1\\nline2"' },
    ]);
  });
});
