"use client";

import { useCallback, useEffect, useState } from "react";
import AuthScreen from "./AuthScreen";
import Chat from "./Chat";
import { getMe, signOut } from "@/lib/api";
import type { User } from "@/lib/types";

/**
 * The auth gate: ask the API who is signed in, then show the sign-in screen or the chat.
 * Signing out unmounts <Chat>, so none of one user's chats stay in memory for the next.
 */
export default function App() {
  const [user, setUser] = useState<User | null | undefined>(undefined); // undefined = still checking
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getMe()
      .then(setUser)
      .catch(() => {
        setOffline(true);
        setUser(null);
      });
  }, []);

  // stable functions, so Chat's effects that use them don't restart on every render
  const handleSignOut = useCallback(() => {
    signOut().catch(() => {}); // the cookie is cleared server-side; leave the UI either way
    setUser(null);
  }, []);
  const handleSessionLost = useCallback(() => setUser(null), []);

  if (user === undefined) return <div className="h-full bg-paper" />;

  if (user === null) {
    return (
      <AuthScreen
        offline={offline}
        onSignedIn={(u) => {
          setOffline(false);
          setUser(u);
        }}
      />
    );
  }

  return (
    <Chat
      key={user.id}
      user={user}
      onSignOut={handleSignOut}
      onSessionLost={handleSessionLost}
    />
  );
}
