
from pathlib import Path

import pymupdf

PDF_PATH = Path(
    "data/papers/attention_is_all_you_need.pdf"
)

doc = pymupdf.open(PDF_PATH)

# Inspect pages 3 through 7 (human-readable page numbers).
for page_index in range(2, min(7, len(doc))):
    page = doc[page_index]

    print(f"\n{'=' * 60}")
    print(f"PAGE {page_index + 1}")
    print("=" * 60)

    blocks = page.get_text("blocks")

    for block in blocks:
        text = block[4].strip().replace("\n", " ")

        if not text:
            continue

        # Print short blocks that might be headings.
        if len(text) < 120:
            print(repr(text))

doc.close()
