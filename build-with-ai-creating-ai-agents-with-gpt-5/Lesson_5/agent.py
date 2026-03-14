"""
Build with AI: Creating AI Agents with GPT-5 (Gemini-only version)

Prereqs:
  pip install python-dotenv fastapi uvicorn requests google-generativeai
  set GEMINI_API_KEY in your environment or .env
"""
# ---------------------------------------------------------------------------
# LESSON 5 (Take your Gemini agent live)
# ---------------------------------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from datetime import datetime
import requests
import os
from google import genai
import re

# Load environment variables
_ = load_dotenv(find_dotenv())

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment or .env file.")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Define tool schema
# ------------------------------
@dataclass
class WeatherInfo:
    city: str
    country: str
    temp_f: float
    condition: str

def get_weather_forecast(city: str):
    """Fetch weather info using the Weather API"""
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        return "Missing WEATHER_API_KEY in environment or .env file."
    WEATHER_BASE_URL = 'https://api.weatherapi.com/v1/current.json'

    try:
        today = datetime.today().strftime('%Y-%m-%d')
        params = {"q": city, "aqi": "no", "key": API_KEY}
        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()
        if "location" not in data or "current" not in data:
            return f"Could not retrieve weather for '{city}'. Try a more specific place name."

        weather = WeatherInfo(
            city=data["location"]["name"],
            country=data["location"]["country"],
            temp_f=float(data["current"]["temp_f"]),
            condition=data["current"]["condition"]["text"]
        )

        return (
            f"Real-time weather report for {today}:\n"
            f"   - City: {weather.city}\n"
            f"   - Country: {weather.country}\n"
            f"   - Temperature: {weather.temp_f:.1f} F\n"
            f"   - Weather Conditions: {weather.condition}"
        )
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

def get_open_meteo_weather():
    """Fetch weather info using Open-Meteo API (no key required)"""
    url = os.getenv("OPEN_METEO_URL")
    if not url:
        return "Missing OPEN_METEO_URL in environment or .env file."
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return f"Open-Meteo weather: {data}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching Open-Meteo data: {str(e)}"

_MODEL_CACHE = None

def _choose_model(client: genai.Client) -> str:
    global _MODEL_CACHE
    if _MODEL_CACHE:
        return _MODEL_CACHE

    preferred = os.getenv("GEMINI_MODEL")
    if preferred:
        _MODEL_CACHE = preferred
        return preferred

    # Prefer a flash model if available; otherwise use the first generateContent-capable model.
    try:
        models = []
        for m in client.models.list():
            if "generateContent" in getattr(m, "supported_actions", []):
                name = getattr(m, "name", "")
                if name.startswith("models/"):
                    name = name.split("/", 1)[1]
                if name:
                    models.append(name)

        flash_models = [m for m in models if "flash" in m.lower()]
        if flash_models:
            _MODEL_CACHE = flash_models[0]
            return _MODEL_CACHE
        if models:
            _MODEL_CACHE = models[0]
            return _MODEL_CACHE
    except Exception:
        pass

    # Fallback to a known example from the official docs.
    _MODEL_CACHE = "gemini-3-flash-preview"
    return _MODEL_CACHE

def gemini_generate_text(prompt: str):
    """Generate text using Gemini API (google-genai)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Missing GEMINI_API_KEY in environment or .env file."
    client = genai.Client(api_key=api_key)
    model_name = _choose_model(client)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return getattr(response, "text", "") or ""
    except Exception as e:
        return f"Gemini API error: {str(e)}"

SYSTEM_INSTRUCTIONS = (
    "You are Trip Coach. You help travelers plan activities to do while on vacation. "
    "If weather info is available, use it first before recommending activities. "
    "Pick activities that solo travelers will enjoy. Be concise and practical."
)

def _needs_weather(prompt: str) -> bool:
    keywords = [
        "weather", "forecast", "temperature", "temp", "rain", "rainy", "snow",
        "sunny", "cloudy", "humid", "wind", "pack", "jacket", "umbrella", "cold", "hot"
    ]
    prompt_lc = prompt.lower()
    return any(k in prompt_lc for k in keywords) or re.search(r"\b(weather|forecast)\b", prompt_lc) is not None

def _extract_city(prompt: str) -> str:
    match = re.search(r"\b(?:in|to|at|around|for)\s+([A-Za-z .'-]+)", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()

# ------------------------------
# Define request model
# ------------------------------
class UserPrompt(BaseModel):
    prompt: str

# ------------------------------
# *******TO RUN THE FASTAPI, FOLLOW THE STEPS BELOW*************
# 
# cd Lesson 5 folder
# make the Port public
# run the command: uvicorn agent:app --reload
# test in Postman or cURL
# POST to a similar endpoint (replace with your endpoint) - https://ideal-space-barnacle-pwqqgv6gq45396x9-8000.app.github.dev/ask
# Send in a JSON request body:
#   {
#      "prompt": "I'm heading to Atlanta this weekend. What's the weather like, and what should I pack?"
#   }
# ------------------------------
@app.post("/ask")
async def ask_agent(request: UserPrompt):
    weather_info = ""
    if _needs_weather(request.prompt):
        if os.getenv("WEATHER_API_KEY"):
            weather_info = get_weather_forecast(_extract_city(request.prompt))
        elif os.getenv("OPEN_METEO_URL"):
            weather_info = get_open_meteo_weather()

    composed_prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"User request:\n{request.prompt}\n\n"
        f"Weather info (if any):\n{weather_info}\n\n"
        "Provide a helpful response."
    )
    response_text = gemini_generate_text(composed_prompt)
    return {"response": response_text}

@app.get("/health")
async def health():
    return {"status": "ok"}

