## Mindwell Therapy Chatbot — Documentation

### What this is
Mindwell is a small web app that provides a **supportive, therapy-style chatbot** experience (CBT- and mindfulness-inspired). It can call an LLM via an API key, and it includes a **built-in crisis safety override** for self-harm language.

### Important safety notes
- **Not medical advice**: Mindwell is not a clinician and does not diagnose or replace professional care.
- **Emergencies**: If a user may be in immediate danger, the app should direct them to local emergency services.
- **Self-harm**: The server performs a simple keyword-based check on the latest user message. If triggered, it returns a crisis response and does not call the LLM.

### Architecture
- **UI**: `src/app/page.tsx` — chat UI (client component)
- **API**: `src/app/api/chat/route.ts` — accepts `{ messages }` and returns `{ reply }`
- **Safety utilities**: `src/lib/safety.ts`
- **LLM caller**: `src/lib/openai.ts` (calls `OPENAI_BASE_URL/chat/completions`)

### Configuration
Create a local env file:
- Copy `.env.example` to `.env.local`
- Set `OPENAI_API_KEY`

The repo includes a `.gitignore` that excludes `.env.local` to help prevent accidentally committing secrets.

Optional:
- `OPENAI_BASE_URL` (OpenAI-compatible providers)
- `OPENAI_MODEL` (defaults to `gpt-4o-mini`)

### Running locally
```bash
npm install
npm run dev
```
Then open `http://localhost:3000`.

### API contract
**POST** `/api/chat`

Request body:
```json
{
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response body:
```json
{ "reply": "..." }
```

Notes:
- Only `user` and `assistant` messages are accepted from the client.
- The API injects its own `system` prompt to enforce Mindwell behavior.
- The API truncates history to the most recent 24 messages.
