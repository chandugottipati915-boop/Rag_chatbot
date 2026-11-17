"use client";

// Chat reads localStorage while rendering, which doesn't exist on the server.
// Loading it with ssr: false renders it only in the browser, so the server
// HTML and the first browser render can't disagree (no hydration errors).
import dynamic from "next/dynamic";

const Chat = dynamic(() => import("./Chat"), {
  ssr: false,
  loading: () => <div className="h-full bg-paper" />,
});

export default function ClientChat() {
  return <Chat />;
}
