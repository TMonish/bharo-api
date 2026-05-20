from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json

app = FastAPI(title="Bharo AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"  # Best free model on Groq

SYSTEM_PROMPT = """You are Bharo, a smart personal AI voice assistant running on an Android phone.
You help the user with tasks by understanding their voice commands.

When the user gives a command, respond with a JSON object like this:
{
  "action": "one of: chat, search, open_app, call, whatsapp, email, instagram",
  "response": "what you say back to the user (friendly, short, conversational)",
  "data": {
    // For search: { "query": "search query" }
    // For open_app: { "package": "com.example.app", "app_name": "YouTube" }
    // For call: { "contact": "Mom" }
    // For whatsapp: { "contact": "John", "message": "I'm on my way" }
    // For email: { "to": "boss@email.com", "subject": "Late", "body": "I'll be late" }
    // For instagram: { "user": "username", "message": "Hey!" }
    // For chat: {}
  }
}

Common Android package names:
- YouTube: com.google.android.youtube
- WhatsApp: com.whatsapp
- Instagram: com.instagram.android
- Gmail: com.google.android.gm
- Chrome: com.android.chrome
- Camera: com.android.camera2
- Settings: com.android.settings
- Spotify: com.spotify.music
- Maps: com.google.android.apps.maps
- Calculator: com.android.calculator2
- Twitter/X: com.twitter.android
- Facebook: com.facebook.katana
- Telegram: org.telegram.messenger
- Netflix: com.netflix.mediaclient
- Snapchat: com.snapchat.android

Always respond ONLY with valid JSON. Be friendly and brief in the "response" field."""


class CommandRequest(BaseModel):
    text: str
    conversation_history: list = []


class SearchRequest(BaseModel):
    query: str


async def call_groq(messages: list, max_tokens: int = 1000) -> str:
    """Call the Groq API and return the response text"""
    async with httpx.AsyncClient() as http:
        response = await http.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=30
        )
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


@app.get("/")
def root():
    return {"status": "Bharo AI is running 🚀", "version": "2.0", "engine": "Groq (Free)"}


@app.post("/command")
async def process_command(req: CommandRequest):
    """Main endpoint: processes a voice command from the Android app"""
    try:
        # Build messages for Groq
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        for msg in req.conversation_history:
            messages.append(msg)

        # Add current user message
        messages.append({"role": "user", "content": req.text})

        raw = await call_groq(messages)

        # Parse the JSON response
        try:
            # Sometimes model wraps in ```json ... ```
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {
                "action": "chat",
                "response": raw,
                "data": {}
            }

        return {
            "success": True,
            "action": result.get("action", "chat"),
            "response": result.get("response", "Done!"),
            "data": result.get("data", {}),
            "assistant_message": {"role": "assistant", "content": raw}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def web_search(req: SearchRequest):
    """Search the web using DuckDuckGo and summarise with Groq"""
    try:
        # Search DuckDuckGo
        search_url = f"https://api.duckduckgo.com/?q={req.query}&format=json&no_html=1"
        async with httpx.AsyncClient() as http:
            resp = await http.get(search_url, timeout=10)
            data = resp.json()

        abstract = data.get("AbstractText", "")
        related = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:3] if "Text" in r]
        raw_info = abstract or " | ".join(related) or "No direct results found."

        # Ask Groq to summarise
        summary = await call_groq([
            {
                "role": "user",
                "content": f"Summarise this in 2-3 sentences for a voice assistant response about '{req.query}': {raw_info}"
            }
        ], max_tokens=200)

        return {
            "success": True,
            "query": req.query,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "engine": "Groq"}
