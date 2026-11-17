"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Composer from "./Composer";
import EmptyState from "./EmptyState";
import { AssistantMessage, UserMessage } from "./Message";
import Sidebar from "./Sidebar";
import SourceList from "./SourceList";
import { CloseIcon, MenuIcon } from "./icons";
import { ApiError, getHealth, getPapers, streamChat } from "@/lib/api";
import { toHistory } from "@/lib/citations";
import { loadConversations, saveConversations } from "@/lib/storage";
import type { ChatMessage, Conversation, Health, Paper } from "@/lib/types";

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function newConversation(question: string): Conversation {
  const now = Date.now();
  return {
    id: newId(),
    title: question.length > 60 ? `${question.slice(0, 57)}…` : question,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

const NO_MESSAGES: ChatMessage[] = [];

/** Which answer's sources fill the margin, and which citation is open. */
type Focus = { messageId: string; n: number | null } | null;

export default function Chat() {
  // ---- state -------------------------------------------------------------
  // Rendered only in the browser (see ClientChat), so localStorage is safe here.
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [focus, setFocus] = useState<Focus>(null);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [paperId, setPaperId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null | undefined>(undefined); // undefined = still checking
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [prefill, setPrefill] = useState<{ text: string; nonce: number } | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const active = conversations.find((c) => c.id === activeId) ?? null;
  const messages = active?.messages ?? NO_MESSAGES;

  // The margin shows the focused answer's sources, or else the latest answer's.
  const lastWithSources = [...messages].reverse().find((m) => m.role === "assistant" && m.sources?.length);
  // (a refusal has no sources, so the margin keeps the previous answer's)
  const focusTarget = focus ? messages.find((m) => m.id === focus.messageId) : undefined;
  const focusedMessage = (focusTarget?.sources?.length ? focusTarget : lastWithSources) ?? null;
  const activeN = focus && focusedMessage && focus.messageId === focusedMessage.id ? focus.n : null;

  // ---- effects -----------------------------------------------------------
  // Save history whenever it changes, but not on every streamed token.
  useEffect(() => {
    if (!busy) saveConversations(conversations);
  }, [conversations, busy]);

  // The API's health and the paper list, for the status dot and the filter.
  useEffect(() => {
    let cancelled = false;
    const check = () =>
      getHealth().then((h) => {
        if (cancelled) return;
        setHealth(h);
        if (h) getPapers().then((p) => !cancelled && setPapers(p)).catch(() => {});
      });
    check();
    const timer = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // Follow the streaming answer, unless the reader has scrolled up.
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // ---- helpers -----------------------------------------------------------
  const patchMessage = useCallback(
    (convId: string, msgId: string, change: (m: ChatMessage) => ChatMessage) =>
      setConversations((cs) =>
        cs.map((c) =>
          c.id !== convId
            ? c
            : { ...c, updatedAt: Date.now(), messages: c.messages.map((m) => (m.id === msgId ? change(m) : m)) },
        ),
      ),
    [],
  );

  // ---- actions -----------------------------------------------------------
  async function send(question: string) {
    if (busy) return;

    const conv = active ?? newConversation(question);
    const history = toHistory(conv.messages); // earlier turns only
    const userMsg: ChatMessage = { id: newId(), role: "user", content: question };
    const reply: ChatMessage = { id: newId(), role: "assistant", content: "", status: "streaming" };

    setConversations((cs) => [
      { ...conv, updatedAt: Date.now(), messages: [...conv.messages, userMsg, reply] },
      ...cs.filter((c) => c.id !== conv.id),
    ]);
    setActiveId(conv.id);
    setFocus({ messageId: reply.id, n: null });
    stickToBottom.current = true;

    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);

    try {
      await streamChat(
        { question, top_k: 5, arxiv_id: paperId, history },
        {
          onSources: (sources) => patchMessage(conv.id, reply.id, (m) => ({ ...m, sources })),
          onToken: (t) => patchMessage(conv.id, reply.id, (m) => ({ ...m, content: m.content + t })),
          onDone: (meta) => patchMessage(conv.id, reply.id, (m) => ({ ...m, meta, status: "done" })),
        },
        controller.signal,
      );
      // the stream ended without a `done` event: keep what arrived
      patchMessage(conv.id, reply.id, (m) => (m.status === "streaming" ? { ...m, status: "done" } : m));
    } catch (e) {
      if (controller.signal.aborted) {
        patchMessage(conv.id, reply.id, (m) => ({ ...m, status: "stopped" }));
      } else {
        const error = e instanceof ApiError ? e.message : "Something went wrong reading the answer. Try again.";
        patchMessage(conv.id, reply.id, (m) => ({ ...m, status: "error", error }));
        getHealth().then(setHealth);
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  function startNewChat() {
    if (busy) abortRef.current?.abort();
    setActiveId(null);
    setFocus(null);
    setSidebarOpen(false);
  }

  function deleteChat(id: string) {
    if (id === activeId) startNewChat();
    setConversations((cs) => cs.filter((c) => c.id !== id));
  }

  function selectChat(id: string) {
    if (busy) abortRef.current?.abort();
    setActiveId(id);
    setFocus(null);
    setSidebarOpen(false);
    stickToBottom.current = true;
  }

  // ---- layout ------------------------------------------------------------
  //  [ sidebar ] [ conversation column ] [ margin: cited passages (lg+) ]
  const showMargin = !!focusedMessage?.sources?.length;

  return (
    <div className="flex h-full">
      {/* sidebar: fixed column on desktop, slide-over on mobile */}
      <aside
        className={`fixed inset-y-0 left-0 z-30 w-72 border-r border-rule bg-paper transition-transform md:static md:w-64 md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          health={health}
          onSelect={selectChat}
          onNew={startNewChat}
          onDelete={deleteChat}
        />
      </aside>
      {sidebarOpen && (
        <button
          aria-label="Close the conversation list"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 z-20 bg-ink/20 md:hidden"
        />
      )}

      <main className="flex min-w-0 flex-1 flex-col bg-sheet">
        <header className="flex items-center gap-2 border-b border-rule px-3 py-2 md:hidden">
          <button
            onClick={() => setSidebarOpen(true)}
            aria-label="Open the conversation list"
            className="rounded p-2 text-ink hover:bg-paper"
          >
            <MenuIcon className="h-5 w-5" />
          </button>
          <span className="font-serif font-semibold">Paper Reader</span>
        </header>

        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
          className="flex-1 overflow-y-auto"
        >
          {messages.length === 0 ? (
            <EmptyState health={health} papers={papers} onPick={(text) => setPrefill({ text, nonce: Date.now() })} />
          ) : (
            <div className="mx-auto w-full max-w-[42rem] space-y-8 px-5 py-8">
              {messages.map((m) =>
                m.role === "user" ? (
                  <UserMessage key={m.id} message={m} />
                ) : (
                  <AssistantMessage
                    key={m.id}
                    message={m}
                    focused={focusedMessage?.id === m.id}
                    activeN={focus?.messageId === m.id ? focus.n : null}
                    onCite={(n) => setFocus({ messageId: m.id, n })}
                    onFocus={() => setFocus({ messageId: m.id, n: null })}
                  />
                ),
              )}
            </div>
          )}
        </div>

        <div className="mx-auto w-full max-w-[42rem] px-5 pt-2 pb-5">
          <Composer
            busy={busy}
            papers={papers}
            paperId={paperId}
            onPaperChange={setPaperId}
            onSend={send}
            onStop={() => abortRef.current?.abort()}
            prefill={prefill}
          />
          <p className="mt-2 text-center text-[0.7rem] text-muted">
            Answers can be wrong. Check the cited passage before relying on one.
          </p>
        </div>
      </main>

      {/* the margin: where cited passages are read */}
      {showMargin && focusedMessage && (
        <aside aria-label="Sources" className="hidden w-[22rem] shrink-0 flex-col border-l border-rule bg-paper lg:flex">
          <div className="flex items-center justify-between px-4 pt-5 pb-3">
            <p className="text-sm font-medium text-ink">Sources for this answer</p>
            {activeN != null && (
              <button
                onClick={() => setFocus({ messageId: focusedMessage.id, n: null })}
                className="flex items-center gap-1 text-xs text-muted hover:text-ink"
              >
                <CloseIcon className="h-3 w-3" /> Close passage
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto px-4 pb-6">
            <SourceList
              sources={focusedMessage.sources!}
              activeN={activeN}
              onSelect={(n) => setFocus({ messageId: focusedMessage.id, n })}
              idPrefix="margin"
            />
          </div>
        </aside>
      )}
    </div>
  );
}
