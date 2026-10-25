
from pathlib import Path

from paperpilot.ingest import parse_pdf


pdf_path = Path(
    "data/papers/attention_is_all_you_need.pdf"
)

paper = parse_pdf(pdf_path)

print("=" * 60)
print("PAPER INFORMATION")
print("=" * 60)

print("Paper ID:", paper.paper_id)
print("Title:", paper.title)
print("Total paragraphs:", len(paper.paragraphs))

sections = sorted({
    paragraph.section
    for paragraph in paper.paragraphs
})

print("\nDetected sections:")
for section in sections:
    print("-", section)

print("\nFIRST 5 PARAGRAPHS")
print("=" * 60)

for index, paragraph in enumerate(paper.paragraphs[:5], start=1):
    print(f"\nParagraph {index}")
    print("Section:", paragraph.section)
    print("Page:", paragraph.page)
    print("Text:", paragraph.text[:400])
    print("-" * 60)
