import google.generativeai as genai
from modules.secrets_manager import load_api_key
import os

key = load_api_key()
if not key:
    print("No API Key found in secrets.json")
else:
    print(f"API Key found (starts with {key[:4]}...)")
    try:
        genai.configure(api_key=key)
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
