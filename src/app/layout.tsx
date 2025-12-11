import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Mindwell — Therapy Chatbot",
  description: "A mindful, supportive therapy-style chatbot powered by an LLM."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
