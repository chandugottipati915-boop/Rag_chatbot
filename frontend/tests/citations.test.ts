import { describe, expect, it } from "vitest";
import { linkCitations, toHistory } from "../lib/citations";

describe("linkCitations", () => {
  it("links single and grouped citations", () => {
    expect(linkCitations("RAG [1] and Self-RAG [2, 3].")).toBe(
      "RAG [1](#cite-1) and Self-RAG [2](#cite-2)[3](#cite-3).",
    );
  });

  it("leaves real markdown links and non-numeric brackets alone", () => {
    expect(linkCitations("see [paper](https://arxiv.org) and [MASK]")).toBe(
      "see [paper](https://arxiv.org) and [MASK]",
    );
  });
});

describe("toHistory", () => {
  it("keeps finished turns, drops failed and empty replies, caps at 20", () => {
    const msgs = [
      { role: "user" as const, content: "q1" },
      { role: "assistant" as const, content: "a1", status: "done" },
      { role: "user" as const, content: "q2" },
      { role: "assistant" as const, content: "partial", status: "error" },
      { role: "assistant" as const, content: "", status: "done" },
    ];
    expect(toHistory(msgs)).toEqual([
      { role: "user", content: "q1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "q2" },
    ]);
    const many = Array.from({ length: 30 }, (_, i) => ({ role: "user" as const, content: `q${i}` }));
    expect(toHistory(many)).toHaveLength(20);
    expect(toHistory(many)[0].content).toBe("q10");
  });
});
