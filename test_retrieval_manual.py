
from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve


store = PaperVectorStore(
    persist_directory="data/chroma",
)

question = (
    "What positional encodings are used "
    "in the Transformer?"
)

results = retrieve(
    store,
    question,
    top_k=5,
    candidate_k=15,
)

print("\nQUESTION:", question)
print("\nRERANKED RESULTS")
print("=" * 60)

for index, result in enumerate(results, start=1):
    print(f"\nRESULT {index}")
    print("Chunk ID:", result["chunk_id"])
    print("Page:", result["metadata"]["page"])
    print("Semantic score:", round(result["semantic_score"], 4))
    print("Keyword score:", round(result["keyword_score"], 4))
    print("Combined score:", round(result["combined_score"], 4))
    print("Text:", result["text"][:600])
    print("-" * 60)
