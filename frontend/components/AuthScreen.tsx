"use client";

import { useState } from "react";
import { API_URL, ApiError, QUESTION_REJECTED, signIn, signUp } from "@/lib/api";
import type { User } from "@/lib/types";

type Mode = "signin" | "signup";

/** Sign in or create an account. The API sets an httpOnly session cookie on success. */
export default function AuthScreen({
  onSignedIn,
  offline,
}: {
  onSignedIn: (user: User) => void;
  offline: boolean;
}) {
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const user = mode === "signin" ? await signIn(email, password) : await signUp(email, password);
      onSignedIn(user);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message === QUESTION_REJECTED                     // a 422: the API's validation
            ? "Enter a valid email and a password of at least 8 characters."
            : err.message
          : "Something went wrong. Try again.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  function switchTo(next: Mode) {
    setMode(next);
    setError(null);
  }

  return (
    <main className="flex min-h-full items-center justify-center bg-paper px-5 py-12">
      <div className="w-full max-w-sm">
        <p className="font-serif text-[2rem] leading-tight font-semibold tracking-tight text-ink">Paper Reader</p>
        <p className="mt-2 text-[0.95rem] leading-relaxed text-muted">
          Cited answers from 15 papers on retrieval-augmented generation. Your chats are saved to your account.
        </p>

        <div className="mt-8 rounded-lg border border-rule bg-sheet p-6">
          <h1 className="font-serif text-xl font-semibold text-ink">
            {mode === "signin" ? "Sign in" : "Create an account"}
          </h1>

          {offline && (
            <p role="alert" className="mt-4 rounded-md bg-danger/8 px-3 py-2 text-sm text-danger">
              Can&apos;t reach the API at {API_URL}. Start it, then reload this page.
            </p>
          )}

          <form onSubmit={submit} className="mt-5 space-y-4">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-ink">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1.5 block w-full rounded-md border border-rule bg-sheet px-3 py-2 text-[0.95rem] text-ink focus:border-teal focus:outline-none"
              />
            </div>
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-ink">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete={mode === "signin" ? "current-password" : "new-password"}
                required
                minLength={8}
                maxLength={128}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                aria-describedby={mode === "signup" ? "password-hint" : undefined}
                className="mt-1.5 block w-full rounded-md border border-rule bg-sheet px-3 py-2 text-[0.95rem] text-ink focus:border-teal focus:outline-none"
              />
              {mode === "signup" && (
                <p id="password-hint" className="mt-1.5 text-xs text-muted">
                  At least 8 characters. A short phrase is easier to remember than symbols.
                </p>
              )}
            </div>

            {error && (
              <p role="alert" className="text-sm text-danger">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="h-10 w-full rounded-md bg-teal text-sm font-medium text-white hover:bg-teal-dark disabled:opacity-50"
            >
              {submitting
                ? mode === "signin"
                  ? "Signing in…"
                  : "Creating your account…"
                : mode === "signin"
                  ? "Sign in"
                  : "Create account"}
            </button>
          </form>
        </div>

        <p className="mt-5 text-center text-sm text-muted">
          {mode === "signin" ? (
            <>
              New here?{" "}
              <button onClick={() => switchTo("signup")} className="font-medium text-teal hover:underline">
                Create an account
              </button>
            </>
          ) : (
            <>
              Have an account?{" "}
              <button onClick={() => switchTo("signin")} className="font-medium text-teal hover:underline">
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </main>
  );
}
