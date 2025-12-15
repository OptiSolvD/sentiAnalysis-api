from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
import numpy as np

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="Emotion Analysis API")

# -------------------------
# Load model (lightweight)
# -------------------------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
emotion_analyzer = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None
)

# -------------------------
# Request schema
# -------------------------
class TextInput(BaseModel):
    text: str

# -------------------------
# Helper: chunk long text
# -------------------------
def chunk_text(text, max_tokens=450):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]
    return [tokenizer.decode(chunk) for chunk in chunks]

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/analyze-emotion")
def analyze_emotion(data: TextInput):
    chunks = chunk_text(data.text)

    all_scores = []

    for chunk in chunks:
        result = emotion_analyzer(chunk)[0]
        all_scores.append({r["label"]: r["score"] for r in result})

    # Average scores across chunks
    final_emotions = {}
    for label in all_scores[0].keys():
        final_emotions[label] = float(
            np.mean([score[label] for score in all_scores])
        )

    # Sort by intensity
    final_emotions = dict(
        sorted(final_emotions.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "dominant_emotion": list(final_emotions.keys())[0],
        "emotion_scores": final_emotions
    }
