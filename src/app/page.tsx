"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  role: Role;
  content: string;
};

const DEFAULT_MESSAGES: ChatMessage[] = [
  {
    role: "assistant",
    content:
      "Hi — I’m Mindwell. I can offer supportive, therapy-style conversation (CBT- and mindfulness-inspired).\n\nWhat’s been weighing on you lately?"
  }
];

export default function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>(DEFAULT_MESSAGES);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [consent, setConsent] = useState(false);
  const chatRef = useRef<HTMLDivElement | null>(null);

  const canSend = useMemo(() => {
    return consent && !isSending && input.trim().length > 0;
  }, [consent, isSending, input]);

  useEffect(() => {
    const el = chatRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, isSending]);

  async function send() {
    const text = input.trim();
    if (!text || isSending || !consent) return;

    setInput("");
    setIsSending(true);

    const nextMessages = [...messages, { role: "user", content: text } as const];
    setMessages(nextMessages);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: nextMessages })
      });

      const data = (await res.json()) as { reply?: string; error?: string };

      if (!res.ok) {
        throw new Error(data.error || "Request failed");
      }

      setMessages((prev) => [...prev, { role: "assistant", content: data.reply ?? "" }]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I ran into an error generating a response. If you set an LLM key, make sure it’s configured correctly.\n\nError: " +
            msg
        }
      ]);
    } finally {
      setIsSending(false);
    }
  }

  function reset() {
    setMessages(DEFAULT_MESSAGES);
    setInput("");
  }

  return (
    <main className="container">
      <div className="header">
        <div className="brand">
          <div className="title">Mindwell</div>
          <div className="subtitle">Supportive therapy-style chatbot (LLM-powered)</div>
        </div>
        <div className="buttonRow">
          <button className="button buttonDanger" onClick={reset} type="button">
            Reset chat
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="notice">
          <strong>Important:</strong> Mindwell is not a medical professional and can’t provide diagnosis or emergency help.
          If you might be in danger or considering self-harm, call your local emergency number now or contact a crisis
          hotline in your country.
          <div style={{ marginTop: 8 }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
              />
              <span>I understand and want to continue.</span>
            </label>
          </div>
        </div>

        <div className="chat" ref={chatRef} aria-live="polite">
          {messages.map((m, idx) => (
            <div key={idx} className={`row ${m.role}`}>
              <div className="bubble">{m.content}</div>
            </div>
          ))}
          {isSending ? (
            <div className="row assistant">
              <div className="bubble">Thinking…</div>
            </div>
          ) : null}
        </div>

        <div className="controls">
          <input
            className="input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send();
              }
            }}
            placeholder={consent ? "Type a message… (Enter to send)" : "Check the box above to enable chat"}
            disabled={!consent}
          />
          <div className="buttonRow">
            <button
              className={`button buttonPrimary`}
              onClick={() => void send()}
              disabled={!canSend}
              type="button"
              title="Send (Enter)"
            >
              Send
            </button>
          </div>
        </div>
      </div>

      <div className="footer">
        <div className="small">
          Tips: Press <span className="kbd">Enter</span> to send. If no API key is set, the app uses a safe local demo
          mode.
        </div>
      </div>
    </main>
  );
}
