"""
Build with AI: Creating AI Agents with Gemini

Prereqs:
  pip install python-dotenv requests google-genai
  set GEMINI_API_KEY in your environment or .env
"""
import os
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass
from datetime import datetime
import requests
from google import genai


# read local .env file
_ = load_dotenv(find_dotenv()) 

# Ensure required keys are present early.
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY in environment or .env file.")

# # ---------------------------------------------------------------------------
# # LESSON 4 (Extend the agent with memory)
# # ---------------------------------------------------------------------------
@dataclass
class WeatherInfo:
    city: str
    country: str
    temp_f: float
    condition: str

def get_weather_forecast(city: str):
    """Fetch weather info using the Weather API - https://www.weatherapi.com/
       Create an account and generate your API key - https://www.weatherapi.com/my/ 
    """
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        return "Missing WEATHER_API_KEY in environment or .env file."
    WEATHER_BASE_URL = 'https://api.weatherapi.com/v1/current.json'

    try:
        today = datetime.today().strftime('%Y-%m-%d')
        params = {"q": city, "aqi": "no", "key": API_KEY}
        
        #construct request and call api
        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Basic validation
        if "location" not in data or "current" not in data:
            return f"Could not retrieve weather for '{city}'. Try a more specific place name."

        weather = WeatherInfo(
            city=data["location"]["name"],
            country=data["location"]["country"],
            temp_f=float(data["current"]["temp_f"]),
            condition=data["current"]["condition"]["text"]
        )

        weather_report = [f"Real-time weather report for {today}:"]

        weather_report.append(
                f"   - City: {weather.city}"
                f"   - Country: {weather.country}"
                f"   - Temperature: {weather.temp_f:.1f} F"
                f"   - Weather Conditions: {weather.condition}"
            )

        return "\n".join(weather_report)
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

    _MODEL_CACHE = "gemini-3-flash-preview"
    return _MODEL_CACHE

def gemini_generate_text(prompt: str):
    """Generate text using Gemini API (google-genai)"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Missing GEMINI_API_KEY in environment or .env file."
    client = genai.Client(api_key=api_key)
    model_name = _choose_model(client)
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return getattr(response, "text", "") or ""

SYSTEM_INSTRUCTIONS = (
    "You are Trip Coach. You help travelers plan activities to do while on vacation. "
    "Check real-time weather first if available. "
    "Make sure recommendations are good for solo travelers. Be clear and practical."
)

def _compose_prompt(history: list[dict], user_prompt: str, weather_info: str) -> str:
    history_text = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_lines.append(f"{role.title()}: {content}")
        history_text = "\n".join(history_lines)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"User request:\n{user_prompt}\n\n"
        f"Weather info (if any):\n{weather_info}\n\n"
        "Provide a helpful response."
    )

def main() -> None:
    city = "Atlanta"
    history: list[dict] = []

    weather_info = ""
    if os.getenv("WEATHER_API_KEY"):
        weather_info = get_weather_forecast(city)
    elif os.getenv("OPEN_METEO_URL"):
        weather_info = get_open_meteo_weather()

    first_prompt = f"""Headed to {city} today. What's the weather like, and what 
                       should I pack?"""
    composed_prompt = _compose_prompt(history, first_prompt, weather_info)
    first_response = gemini_generate_text(composed_prompt)
    print(first_response)
    print("-" * 70)
    history.extend([
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_response},
    ])

    second_prompt = "Can you recommend a seafood restaurant?"
    composed_prompt = _compose_prompt(history, second_prompt, weather_info)
    second_response = gemini_generate_text(composed_prompt)
    print(second_response)

if __name__ == "__main__":
    main()

