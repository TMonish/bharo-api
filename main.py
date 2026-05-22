from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json

app = FastAPI(title="Bharo AI - Full Jarvis Mode")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama3-70b-8192"

SYSTEM_PROMPT = """You are Bharo, a powerful personal AI assistant like Jarvis from Iron Man.
You run on the user's Android phone and can control everything on it.
You are smart, friendly, and speak in short confident sentences.

When the user gives a command, respond ONLY with a JSON object:
{
  "action": "one of the actions below",
  "response": "short friendly voice response to speak aloud",
  "data": { ... relevant data ... }
}

ACTIONS AND DATA FORMAT:

chat: general conversation
  data: {}

search: search the web
  data: { "query": "search query" }

open_app: open any app
  data: { "package": "com.example.app", "app_name": "YouTube" }

call: make a phone call
  data: { "contact": "contact name or number" }

whatsapp: send whatsapp message
  data: { "contact": "name or number", "message": "message text" }

sms: send SMS
  data: { "contact": "name or number", "message": "message text" }

email: send email
  data: { "to": "email", "subject": "subject", "body": "body" }

alarm: set an alarm
  data: { "hour": 7, "minute": 30, "message": "Wake up" }

flashlight_on: turn on flashlight
  data: {}

flashlight_off: turn off flashlight
  data: {}

volume_up: increase volume
  data: { "steps": 1 }

volume_down: decrease volume
  data: { "steps": 1 }

volume_mute: mute phone
  data: {}

brightness_up: increase brightness
  data: {}

brightness_down: decrease brightness
  data: {}

wifi_on: turn on WiFi
  data: {}

wifi_off: turn off WiFi
  data: {}

bluetooth_on: turn on Bluetooth
  data: {}

bluetooth_off: turn off Bluetooth
  data: {}

take_photo: take a photo
  data: {}

read_notifications: read recent notifications
  data: {}

lock_screen: lock the phone screen
  data: {}

play_music: play music
  data: { "query": "song or artist name", "app": "spotify or youtube" }

weather: get weather
  data: { "city": "city name" }

translate: translate text
  data: { "text": "text to translate", "language": "target language" }

calculate: do a calculation
  data: { "expression": "2 + 2" }

COMMON APP PACKAGES:
YouTube: com.google.android.youtube
WhatsApp: com.whatsapp
Instagram: com.instagram.android
Gmail: com.google.android.gm
Chrome: com.android.chrome
Camera: com.android.camera2
Settings: com.android.settings
Spotify: com.spotify.music
Maps: com.google.android.apps.maps
Calculator: com.android.calculator2
Twitter: com.twitter.android
Facebook: com.facebook.katana
Telegram: org.telegram.messenger
Netflix: com.netflix.mediaclient
Snapchat: com.snapchat.android
Clock: com.android.deskclock
Contacts: com.android.contacts
Gallery: com.android.gallery3d
Files: com.android.documentsui
Play Store: com.android.vending

Always respond ONLY with valid JSON. Keep "response" short, friendly and confident like Jarvis."""


class CommandRequest(BaseModel):
    text: str
    conversation_history: list = []


class SearchRequest(BaseModel):
    query: str


class NotificationRequest(BaseModel):
    notifications: list


async def call_groq(messages: list, max_tokens: int = 1000) -> str:
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
    return {"status": "Bharo AI is running 🚀", "version": "3.0", "mode": "Full Jarvis"}


@app.post("/command")
async def process_command(req: CommandRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in req.conversation_history:
            messages.append(msg)
        messages.append({"role": "user", "content": req.text})

        raw = await call_groq(messages)

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {"action": "chat", "response": raw, "data": {}}

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
    try:
        search_url = f"https://api.duckduckgo.com/?q={req.query}&format=json&no_html=1"
        async with httpx.AsyncClient() as http:
            resp = await http.get(search_url, timeout=10)
            data = resp.json()

        abstract = data.get("AbstractText", "")
        related = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:3] if "Text" in r]
        raw_info = abstract or " | ".join(related) or "No direct results found."

        summary = await call_groq([{
            "role": "user",
            "content": f"Summarise in 2-3 sentences for a voice assistant about '{req.query}': {raw_info}"
        }], max_tokens=200)

        return {"success": True, "query": req.query, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarise_notification")
async def summarise_notification(req: NotificationRequest):
    try:
        notif_text = "\n".join([
            f"App: {n.get('app', 'Unknown')}, From: {n.get('title', 'Unknown')}, Message: {n.get('text', '')}"
            for n in req.notifications
        ])

        summary = await call_groq([{
            "role": "user",
            "content": f"You are Bharo AI assistant. Summarise these notifications naturally as if speaking to the user. Keep it short and friendly:\n{notif_text}"
        }], max_tokens=200)

        return {"success": True, "summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}
