export type ChatRole = "system" | "user" | "assistant";

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export function looksLikeSelfHarm(text: string): boolean {
  const t = text.toLowerCase();

  // Heuristic only—kept intentionally broad.
  const phrases = [
    "kill myself",
    "killing myself",
    "end my life",
    "suicide",
    "suicidal",
    "self harm",
    "self-harm",
    "hurt myself",
    "want to die",
    "don\u2019t want to live",
    "dont want to live"
  ];

  return phrases.some((p) => t.includes(p));
}

export function crisisResponse(): string {
  return (
    "I’m really sorry you’re feeling this way — and I’m glad you told me. I can’t help with anything that would harm you.\n\n" +
    "If you may be in immediate danger, call your local emergency number right now.\n" +
    "If you can, also reach out to someone who can stay with you (a friend, family member, or neighbor).\n\n" +
    "If you’re in the U.S. or Canada: call or text 988 (Suicide & Crisis Lifeline).\n" +
    "If you’re in the U.K. & ROI: Samaritans 116 123.\n" +
    "If you’re elsewhere: tell me your country and I’ll help find a local crisis number.\n\n" +
    "If you’re safe for the moment, can you tell me: are you thinking about harming yourself today, and are you alone right now?"
  );
}

export function mindwellSystemPrompt(): string {
  return [
    "You are Mindwell, a supportive therapy-style chatbot.",
    "Goals: be warm, non-judgmental, validating; use evidence-based skills (CBT, grounding, mindfulness, motivational interviewing).",
    "Boundaries: you are not a clinician; do not diagnose; do not claim certainty; encourage professional help when appropriate.",
    "Safety: If user expresses self-harm/suicidal intent or imminent danger, prioritize crisis guidance and encourage contacting local emergency services; do not provide instructions for self-harm.",
    "Style: short paragraphs, ask 1-2 gentle questions, offer 1 small next step.",
    "Privacy: avoid requesting sensitive identifiers; do not store data."
  ].join("\n");
}
