import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Paper Reader: ask the RAG papers",
  description: "Cited answers from 15 papers on retrieval-augmented generation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="h-full">{children}</body>
    </html>
  );
}
