## Mindwell — LLM Therapy Chatbot Web App

A simple Next.js web app that provides a **supportive, therapy-style chatbot** (CBT- and mindfulness-inspired).

### Safety
- **Not medical advice**: This app is not a substitute for professional care.
- **If you may be in danger or considering self-harm**, call your local emergency number or a crisis hotline.
- The server includes a basic safety override for self-harm language.

### Getting started
```bash
npm install
cp .env.example .env.local
# edit .env.local and set OPENAI_API_KEY
npm run dev
```
Open `http://localhost:3000`.

Note: `.env.local` is gitignored by default.

### Configuration
- **`OPENAI_API_KEY`**: required to enable live LLM responses (otherwise demo mode)
- **`OPENAI_BASE_URL`**: optional (defaults to `https://api.openai.com/v1`)
- **`OPENAI_MODEL`**: optional (defaults to `gpt-4o-mini`)

### Where things live
- **UI**: `src/app/page.tsx`
- **API**: `src/app/api/chat/route.ts`
- **Safety utilities**: `src/lib/safety.ts`
- **LLM client**: `src/lib/openai.ts`

### Docs
See `DOCUMENTATION.md` for more details.
