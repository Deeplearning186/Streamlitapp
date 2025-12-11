import { NextResponse } from "next/server";
import { callOpenAIChat } from "@/lib/openai";
import {
  ChatMessage,
  crisisResponse,
  looksLikeSelfHarm,
  mindwellSystemPrompt
} from "@/lib/safety";

type Body = {
  messages?: Array<{ role?: string; content?: string }>;
};

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as Body;

    const raw = Array.isArray(body.messages) ? body.messages : [];
    const messages: ChatMessage[] = raw
      .filter((m) => typeof m?.role === "string" && typeof m?.content === "string")
      .map((m) => ({ role: m.role as ChatMessage["role"], content: m.content as string }))
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.content.trim().length > 0)
      .slice(-24);

    const lastUser = [...messages].reverse().find((m) => m.role === "user")?.content ?? "";

    if (looksLikeSelfHarm(lastUser)) {
      return NextResponse.json({ reply: crisisResponse() }, { status: 200 });
    }

    const system: ChatMessage = {
      role: "system",
      content: mindwellSystemPrompt()
    };

    const reply = await callOpenAIChat({
      messages: [system, ...messages],
      temperature: 0.7
    });

    return NextResponse.json({ reply }, { status: 200 });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
