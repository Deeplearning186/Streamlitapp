import { ChatMessage } from "@/lib/safety";

type OpenAIChatCompletionResponse = {
  choices?: Array<{ message?: { content?: string } }>;
  error?: { message?: string };
};

export function getOpenAIConfig(): {
  apiKey?: string;
  baseUrl: string;
  model: string;
} {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";
  return { apiKey, baseUrl, model };
}

export async function callOpenAIChat(opts: {
  messages: ChatMessage[];
  temperature?: number;
}): Promise<string> {
  const { apiKey, baseUrl, model } = getOpenAIConfig();

  if (!apiKey) {
    // Local fallback demo mode.
    const lastUser = [...opts.messages].reverse().find((m) => m.role === "user")?.content ?? "";

    if (!lastUser.trim()) {
      return "Tell me what’s on your mind, and we’ll take it one small step at a time.";
    }

    return (
      "(Demo mode — no API key configured.)\n\n" +
      "I hear you. Before we problem-solve, let’s slow down for 10 seconds: unclench your jaw, drop your shoulders, and take one slow breath.\n\n" +
      "What emotion is strongest right now (e.g., anxious, sad, angry, numb) — and where do you feel it in your body?"
    );
  }

  const url = `${baseUrl.replace(/\/$/, "")}/chat/completions`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages: opts.messages,
      temperature: opts.temperature ?? 0.7
    })
  });

  const json = (await res.json()) as OpenAIChatCompletionResponse;

  if (!res.ok) {
    const msg = json.error?.message || `OpenAI request failed (${res.status})`;
    throw new Error(msg);
  }

  const content = json.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("Empty model response");
  }

  return content;
}
