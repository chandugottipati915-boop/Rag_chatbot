# Frontend: Paper Reader

A Next.js chat UI for the RAG API. It streams answers from `POST /chat/stream`,
turns `[n]` citations into clickable marks that open the cited passage, and saves
each user's chats through the API (Postgres). Users sign in with email and password.

## Run it

```bash
# 1. backend (from the project root)
docker start qdrant postgres
uv run alembic upgrade head      # first time, and after pulling new migrations
uv run uvicorn rag.api.main:app --app-dir src --reload

# 2. frontend
cd frontend
npm install
npm run dev          # http://localhost:3000
```

The API URL defaults to `http://localhost:8000`. To change it, copy `.env.example`
to `.env.local` and edit `NEXT_PUBLIC_API_URL`.

## Scripts

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Production build (also type-checks) |
| `npm test` | Unit tests (Vitest): SSE parser, citation linking, history |
| `npm run lint` | ESLint, including React's hooks rules |

## How it's put together

```
app/
  layout.tsx        root HTML, fonts and global CSS (server component)
  page.tsx          renders <ClientChat/> (server component)
  globals.css       design tokens (Tailwind v4 @theme) and answer typography
components/
  ClientChat.tsx    loads the app in the browser only (no server rendering)
  App.tsx           auth gate: asks /auth/me, shows AuthScreen or Chat
  AuthScreen.tsx    sign in / create account
  Chat.tsx          all chat state: conversations, streaming, focus, health
  Message.tsx       user bubble, assistant answer, refusal, metadata line
  Answer.tsx        markdown answer; [n] becomes a clickable citation mark
  SourceList.tsx    the cited passages; the active one is highlighted
  Composer.tsx      question box, paper filter, Ask / Stop
  Sidebar.tsx       conversation list and API status
  EmptyState.tsx    first screen: example questions, paper list, offline help
lib/
  api.ts            fetch wrappers; streamChat() reads the SSE stream
  sse.ts            incremental Server-Sent Events parser
  citations.ts      [n] → link rewriting; history for the API
  conversations.ts  saved-chat shapes from the API → UI shapes
  types.ts          shapes shared with src/rag/api/schemas.py
```

### The streaming flow

1. `Chat.send()` creates the chat on the server if it's new (`POST /conversations`), then adds
   the question and an empty "streaming" answer to it.
2. `streamChat()` POSTs the question, the paper filter and the `conversation_id`. The server
   loads the history from Postgres and saves both turns.
3. The response body is read chunk by chunk; `createSSEParser()` buffers until each
   event's blank line arrives, then dispatches `sources` → `token`… → `done`.
4. Each `token` appends to the answer; `done` adds latency, the rewritten query and retries.
5. **Stop** aborts the fetch with an `AbortController`; the partial answer is kept.

### Signing in

The API sets an httpOnly session cookie; the app never sees the token. Every `fetch` uses
`credentials: "include"` so the browser sends that cookie to the API (another origin: port
8000), and the API allows it with `allow_credentials=True` and an exact `CORS_ORIGINS`. Any 401
(`AuthError`) sends you back to the sign-in screen. Signing out unmounts `Chat`, so one user's
chats never stay in memory for the next.

### Saved chats

The sidebar lists `GET /conversations` (titles only); opening a chat loads its messages with
`GET /conversations/{id}`. If the database is down, new chats still work but live only in the
tab: they send their history from the browser, as before, and the sidebar says chats aren't saved.

`EventSource` can't be used because it only supports GET; `/chat/stream` is a POST.
