"use client";

import { CloseIcon, PlusIcon } from "./icons";
import type { Conversation, Health } from "@/lib/types";

export default function Sidebar({
  conversations,
  activeId,
  health,
  onSelect,
  onNew,
  onDelete,
}: {
  conversations: Conversation[];
  activeId: string | null;
  health: Health | null | undefined;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  return (
    <nav aria-label="Conversations" className="flex h-full flex-col">
      <div className="px-4 pt-5 pb-4">
        <p className="font-serif text-lg font-semibold tracking-tight">Paper Reader</p>
        <p className="text-xs text-muted">Cited answers from RAG research</p>
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center gap-2 rounded-md border border-rule bg-sheet px-3 py-2 text-sm font-medium hover:border-ink/30"
        >
          <PlusIcon /> New chat
        </button>
      </div>

      <ul className="mt-4 flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
        {sorted.length === 0 && <li className="px-2 py-1 text-sm text-muted">Your chats will appear here.</li>}
        {sorted.map((c) => (
          <li key={c.id} className="group relative">
            <button
              type="button"
              onClick={() => onSelect(c.id)}
              aria-current={c.id === activeId ? "page" : undefined}
              className={`block w-full truncate rounded-md py-2 pr-8 pl-2.5 text-left text-sm ${
                c.id === activeId ? "bg-sheet font-medium text-ink shadow-[inset_2px_0_0_var(--color-teal)]" : "text-muted hover:bg-sheet/60 hover:text-ink"
              }`}
            >
              {c.title}
            </button>
            <button
              type="button"
              onClick={() => onDelete(c.id)}
              aria-label={`Delete “${c.title}”`}
              className="absolute top-1/2 right-1.5 -translate-y-1/2 rounded p-1 text-muted opacity-0 group-hover:opacity-100 hover:text-danger focus-visible:opacity-100"
            >
              <CloseIcon className="h-3.5 w-3.5" />
            </button>
          </li>
        ))}
      </ul>

      <div className="flex items-center gap-2 border-t border-rule px-4 py-3 text-xs text-muted">
        <span
          aria-hidden
          className={`h-2 w-2 rounded-full ${
            health === undefined ? "bg-rule" : health?.status === "ok" ? "bg-teal" : health ? "bg-amber-500" : "bg-danger"
          }`}
        />
        {health === undefined
          ? "Checking the API"
          : health === null
            ? "API offline"
            : health.status === "ok"
              ? `${health.points?.toLocaleString()} passages indexed`
              : "Index is empty or Qdrant is down"}
      </div>
    </nav>
  );
}
