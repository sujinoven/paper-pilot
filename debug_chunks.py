
from pathlib import Path

from paperpilot.ingest import parse_pdf
from paperpilot.chunking import chunk_paper


paper = parse_pdf(
    Path("data/papers/attention_is_all_you_need.pdf")
)

chunks = chunk_paper(
    paper,
    chunk_size=220,
    overlap=40,
)

keywords = [
    "positional encoding",
    "positional encodings",
    "sinusoidal",
    "sine",
    "cosine",
]

print("=" * 60)
print("SEARCHING CHUNKS FOR POSITIONAL ENCODING")
print("=" * 60)

matches = []

for chunk in chunks:
    text_lower = chunk.text.lower()

    if any(keyword in text_lower for keyword in keywords):
        matches.append(chunk)

print("Matching chunks:", len(matches))

for chunk in matches:
    print("\nChunk ID:", chunk.chunk_id)
    print("Section:", chunk.section)
    print("Page:", chunk.page)
    print("Words:", len(chunk.text.split()))
    print("Text:", chunk.text[:1000])
    print("-" * 60)
