from sentence_transformers import SentenceTransformer
import json
import pickle
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import wordpunct_tokenize

nltk.download("punkt")

# Loading the pre-trained model
model = SentenceTransformer("all-mpnet-base-v2")

with open("courses.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Generating embeddings using Sentence Transformers
texts = [
    f"{item.get('Title', 'Unknown Course').strip() or 'Unknown Course'} - {item.get('Description', 'No description available').strip() or 'No description available'}"
    for item in data
]

embeddings = model.encode(texts, batch_size=20, show_progress_bar=True)

with open("batched_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("768-dimensional embeddings generated and saved successfully using Sentence Transformers!")
