
from pathlib import Path

from paperpilot.ingest import parse_pdf
from paperpilot.chunking import chunk_paper
from paperpilot.vector_store import PaperVectorStore


PDF_PATH = Path(
    "data/papers/attention_is_all_you_need.pdf"
)

print("=" * 60)
print("PAPERPILOT VECTOR DATABASE")
print("=" * 60)

# Step 1: Parse the PDF.
paper = parse_pdf(PDF_PATH)

print("\nPaper:", paper.title)
print("Paragraphs:", len(paper.paragraphs))

# Step 2: Split the paper into chunks.
chunks = chunk_paper(
    paper,
    chunk_size=220,
    overlap=40,
)

print("Chunks generated:", len(chunks))

# Step 3: Open the vector database.
print("\nLoading embedding model and database...")

store = PaperVectorStore(
    persist_directory="data/chroma",
)

# Step 4: Store the chunks.
print("Generating embeddings and storing chunks...")

store.add_chunks(chunks)

print("Chunks stored:", store.count())

# Step 5: Search using a natural-language question.
question = "What positional encodings are used in the Transformer?"

print("\nQUESTION:", question)

results = store.search(
    question,
    top_k=10,
)

print("\nTOP SEARCH RESULTS")
print("=" * 60)

for index, result in enumerate(results, start=1):
    metadata = result["metadata"]

    print(f"\nRESULT {index}")
    print("Chunk ID:", result["chunk_id"])
    print("Paper:", metadata["title"])
    print("Section:", metadata["section"])
    print("Page:", metadata["page"])
    print("Cosine distance:", round(result["distance"], 4))
    print("Text:", result["text"][:1200])
    print("-" * 60)
