from transformers import pipeline, AutoTokenizer
import numpy as np

model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

tokenizer = AutoTokenizer.from_pretrained(model_name)
emotion_analyzer = pipeline(
    "text-classification",
    model=model_name,
    top_k=None
)

def chunk_text(text, max_tokens=450):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [
        tokens[i:i + max_tokens]
        for i in range(0, len(tokens), max_tokens)
    ]
    return [tokenizer.decode(chunk) for chunk in chunks]

text = """The period of rigorous preparation of JEE Advanced is supposed to be purely academic , but for me it is the most formative time of my life, not because of the syllabi of JEE but because of  a single unexpected friendship.I was living on the 2nd floor of a building with my mom and sister fully focused on clearing IIT JEE examination. The JEE phase reached a critical point during the month of January, where my focus should have been absolute but instead of having it , I lost myself. My entire personality and trajectory were about to be reshaped by a kind soul who just lived below me on the 1st floor of the building.
I first met her in the month of the August when she came to my study room. She was very to know about me. I was washing my face with cold water after evening study and preparing myself for another session of study. I still remember that day clearly. At that time I was introvert but after seeing her I don’t know what happened I went to her and began talking with her. I went on talking I didn’t even know how I should hold myself. She listened to me patiently without interruption. Whatever I say she trust blindly she accepted every bit of my words without questioning. Her own secret dream was to become a model or a celebrity. Her family never supported. But she wwwwas quietly moving. The most crucial moment arrived in January when JEE preparation derailed. I was suffering from deep emotional pain after being cheated by school crush. I shared this situation with her which I told no one else. The feeling I experienced after sharing was one of immediate relief like a dark shadow in intolerable heat. She told me that I was made to achieve greatness in life and a true fairy will be waiting for me. But I realized that she was that fairy that God had gifted me . Her guidance was like emotional first aid.I asked her to take a gift from my side after I got selected in IIT but she silently rejected and told me that she will ask when time permits .Then I realised how a true friend should be.
This unique friendship led to my recovery and opened the door of IIT Gandhinagar. Her consistent trying to her own challenging dream demonstrated to me how one must operate in life. She taught me the meaning of trust , communication and daring to visualize the future which seems impossible. When I spoke to her  , I entered a world far better than our own , a world rooted in genuine connection rather than just formality. I realized that true connection and emotional clarity are as essential as academic focus for achieving complex go"""

chunks = chunk_text(text)

all_scores = []

for chunk in chunks:
    result = emotion_analyzer(chunk)[0]
    all_scores.append({r["label"]: r["score"] for r in result})

# Average emotion scores
final_emotions = {}
for label in all_scores[0].keys():
    final_emotions[label] = np.mean([score[label] for score in all_scores])

# Sort by intensity
final_emotions = dict(sorted(final_emotions.items(), key=lambda x: x[1], reverse=True))

print("Final Emotion Analysis:")
for k, v in final_emotions.items():
    print(f"{k}: {v:.2f}")
