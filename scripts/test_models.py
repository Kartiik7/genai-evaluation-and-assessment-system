"""Test all flash model variants for availability."""
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

prompt = 'Return only valid JSON: {"status": "ok"}'

candidates = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

for model_name in candidates:
    try:
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        print(f"OK  {model_name} -> {resp.text[:60]}")
    except Exception as e:
        err = str(e)[:150]
        print(f"ERR {model_name}: {err}")
