
import google.generativeai as genai
import os

API_KEY = "AIzaSyD0On4UCCdbkYCvDkOd18M9H1Wq5fLotXM"
genai.configure(api_key=API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error: {e}")
