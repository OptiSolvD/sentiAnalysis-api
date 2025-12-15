from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer
import numpy as np

# -----------------------
# App initialization
# -----------------------
app = FastAPI(title="Emotion Analysis API")

# -----------------------
# Model setup (runs ONCE)
# -----------------------
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

tokenizer = AutoTokenizer.from_pretrained(model_name)

emotion_analyzer = pipeline(
    "text-classification",
    model=model_name,
    top_k=None,
    truncation=True
)

# -----------------------
# Helper: chunk text
# -----------------------
def chunk_text(text, max_tokens=400):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]
    return [tokenizer.decode(chunk) for chunk in chunks]

# -----------------------
# Request schema
# -----------------------
class TextInput(BaseModel):
    text: str

# -----------------------
# Health check
# -----------------------
@app.get("/")
def health():
    return {"status": "Emotion API running"}

# -----------------------
# Emotion endpoint
# -----------------------
@app.post("/analyze-emotion")
def analyze_emotion(data: TextInput):
    chunks = chunk_text(data.text)
    all_scores = []

    for chunk in chunks:
        result = emotion_analyzer(chunk)[0]
        all_scores.append({r["label"]: r["score"] for r in result})

    final_emotions = {}
    for label in all_scores[0]:
        final_emotions[label] = float(
            np.mean([s[label] for s in all_scores])
        )

    final_emotions = dict(
        sorted(final_emotions.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "top_emotion": list(final_emotions.keys())[0],
        "emotions": final_emotions
    }
