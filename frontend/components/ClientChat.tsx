"use client";

// The app is entirely browser state: it checks who is signed in, fetches the saved
// chats and streams answers after the page loads. Loading it with ssr: false skips rendering it on the
// server, so the server HTML and the first browser render can't disagree.
import dynamic from "next/dynamic";

const App = dynamic(() => import("./App"), {
  ssr: false,
  loading: () => <div className="h-full bg-paper" />,
});

export default function ClientChat() {
  return <App />;
}
