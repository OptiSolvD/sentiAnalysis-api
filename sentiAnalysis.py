from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Emotion Analysis API")

HF_API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

class TextInput(BaseModel):
    text: str

@app.post("/analyze-emotion")
def analyze_emotion(data: TextInput):
    payload = {
        "inputs": data.text
    }

    response = requests.post(HF_API_URL, headers=headers, json=payload)

    # ✅ ADD THIS BLOCK
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "Hugging Face API error",
                "hf_status_code": response.status_code,
                "hf_response": response.text
            }
        )

    results = response.json()[0]

    emotions = {item["label"]: item["score"] for item in results}
    dominant = max(emotions, key=emotions.get)

    return {
        "dominant_emotion": dominant,
        "emotion_scores": emotions
    }

