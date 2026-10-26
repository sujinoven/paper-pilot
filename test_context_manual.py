from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve
from paperpilot.context import format_context


store = PaperVectorStore()

question = (
    "What positional encodings are used in the Transformer?"
)

results = retrieve(
    store,
    question,
    top_k=3,
    candidate_k=15,
)

context = format_context(results)

print("QUESTION:", question)
print("\nFORMATTED EVIDENCE")
print("=" * 60)
print(context)