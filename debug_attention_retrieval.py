from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve


question = "Why does the Transformer use multi-head attention?"

store = PaperVectorStore()

print("\nVECTOR SEARCH — TOP 15")
print("=" * 70)

candidates = store.search(
    question,
    top_k=15,
)

for rank, result in enumerate(candidates, start=1):
    metadata = result["metadata"]

    print(
        f"{rank:2}. "
        f"{metadata['section']} | "
        f"Page {metadata['page']} | "
        f"Distance {result['distance']:.4f}"
    )


print("\nRERANKED SEARCH — TOP 10")
print("=" * 70)

results = retrieve(
    store,
    question,
    top_k=10,
    candidate_k=15,
)

for rank, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(
    f"{rank:2}. "
    f"{metadata['section']} | "
    f"Page {metadata['page']} | "
    f"Semantic {result['semantic_score']:.4f} | "
    f"Keyword {result['keyword_score']:.4f} | "
    f"Section {result['section_score']:.4f} | "
    f"Combined {result['combined_score']:.4f}"
)