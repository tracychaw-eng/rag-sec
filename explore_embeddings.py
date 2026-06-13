import os
from openai import OpenAI
import math

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a**2 for a in vec_a))
    mag_b = math.sqrt(sum(b**2 for b in vec_b))
    return dot / (mag_a * mag_b)

def label_score(score, model="3-small"):
    if model == "ada-002":
        # ada-002 compresses scores into a high narrow range
        return "SIMILAR ✅" if score > 0.85 else ("RELATED" if score > 0.75 else "UNRELATED ❌")
    else:
        # 3-small uses a wider dynamic range
        return "SIMILAR ✅" if score > 0.45 else ("RELATED" if score > 0.20 else "UNRELATED ❌")

# ── Experiment 1: What does an embedding look like? ──
text = "What are the main risk factors?"
embedding = get_embedding(text)

print(f"Text: '{text}'")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {[round(x, 4) for x in embedding[:10]]}")
print(f"Min value: {min(embedding):.4f}")
print(f"Max value: {max(embedding):.4f}")

# ── Experiment 2: Similar questions → similar embeddings ──
print("\n--- SIMILARITY EXPERIMENT ---")
pairs = [
    # Similar pairs — should score HIGH
    ("What are the main risk factors?",
     "What risks does the company face?"),
    ("What cybersecurity risks are mentioned?",
     "How is the company exposed to hacking or data breaches?"),

    # Dissimilar pairs — should score LOW
    ("What are the main risk factors?",
     "What was the company's revenue last year?"),
    ("Cybersecurity threats to the business",
     "The history of Renaissance painting"),
]

for text_a, text_b in pairs:
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    score = cosine_similarity(emb_a, emb_b)
    label = label_score(score, model="3-small")
    print(f"\nScore: {score:.3f}  [{label}]")
    print(f"  A: {text_a}")
    print(f"  B: {text_b}")