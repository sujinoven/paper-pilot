
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

sentences = [
    "The Transformer uses self-attention.",
    "Attention mechanisms help model token relationships.",
    "The weather is sunny today.",
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True,
)

print("\nNumber of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)
print("Vector dimensions:", embeddings.shape[1])

for i, sentence in enumerate(sentences):
    print(f"\nSentence {i + 1}: {sentence}")
    print("First five values:", embeddings[i][:5])

print("\nEMBEDDING COMPARISON")

print(
    "Sentence 1 vs Sentence 2:",
    np.array_equal(embeddings[0], embeddings[1]),
)

print(
    "Sentence 1 vs Sentence 3:",
    np.array_equal(embeddings[0], embeddings[2]),
)

similarity = embeddings @ embeddings.T

print("\nCOSINE SIMILARITY MATRIX")
print(similarity.round(4))
