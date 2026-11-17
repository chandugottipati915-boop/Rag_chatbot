// The model cites sources as [1], [2] or [1, 3]. Rewrite each number into a
// markdown link "#cite-n" so the markdown renderer hands it to our
// CitationMark component instead of printing plain brackets.
export function linkCitations(text: string): string {
  return text.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (_, nums: string) =>
    nums
      .split(",")
      .map((n) => `[${n.trim()}](#cite-${n.trim()})`)
      .join(""),
  );
}

/** Build the history the API expects from finished turns: oldest first, max 20. */
export function toHistory<T extends { role: "user" | "assistant"; content: string; status?: string }>(
  messages: T[],
) {
  return messages
    .filter((m) => m.content.trim() !== "" && (m.role === "user" || m.status === "done"))
    .slice(-20)
    .map((m) => ({ role: m.role, content: m.content.slice(0, 8000) }));
}
