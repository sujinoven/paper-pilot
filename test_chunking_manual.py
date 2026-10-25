
from pathlib import Path
from statistics import mean

from paperpilot.ingest import parse_pdf
from paperpilot.chunking import chunk_paper


pdf_path = Path(
    "data/papers/attention_is_all_you_need.pdf"
)

paper = parse_pdf(pdf_path)

chunks = chunk_paper(
    paper,
    chunk_size=220,
    overlap=40,
)

print("=" * 60)
print("PAPER CHUNKING REPORT")
print("=" * 60)

print("Title:", paper.title)
print("Paper ID:", paper.paper_id)
print("Original paragraphs:", len(paper.paragraphs))
print("Total chunks:", len(chunks))

if chunks:
    sizes = [len(chunk.text.split()) for chunk in chunks]

    print("Minimum chunk size:", min(sizes))
    print("Maximum chunk size:", max(sizes))
    print("Average chunk size:", round(mean(sizes), 2))

print("\nFIRST THREE CHUNKS")
print("=" * 60)

for chunk in chunks[:3]:
    print("\nChunk ID:", chunk.chunk_id)
    print("Section:", chunk.section)
    print("Page:", chunk.page)
    print("Words:", len(chunk.text.split()))
    print("Text:", chunk.text[:600])
    print("-" * 60)
