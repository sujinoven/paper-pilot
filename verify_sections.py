
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

print("Paper:", paper.title)
print("Paragraphs:", len(paper.paragraphs))
print("Chunks:", len(chunks))

print("\nSECTIONS FOUND")
for section in dict.fromkeys(p.section for p in paper.paragraphs):
    print("-", section)

print("\nPOSITIONAL ENCODING CHUNKS")
for chunk in chunks:
    if "positional encod" in chunk.text.lower():
        print("\nChunk:", chunk.chunk_id)
        print("Section:", chunk.section)
        print("Page:", chunk.page)
        print("Text:", chunk.text[:350])
