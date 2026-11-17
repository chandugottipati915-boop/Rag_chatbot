"use client";

// Chat is entirely browser state: it fetches the saved chats and streams answers
// after the page loads. Loading it with ssr: false skips rendering it on the
// server, so the server HTML and the first browser render can't disagree.
import dynamic from "next/dynamic";

const Chat = dynamic(() => import("./Chat"), {
  ssr: false,
  loading: () => <div className="h-full bg-paper" />,
});

export default function ClientChat() {
  return <Chat />;
}
