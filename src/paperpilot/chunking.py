
from dataclasses import dataclass

from .ingest import Paragraph, ParsedPaper


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    text: str


def split_long_paragraph(
    paragraph: Paragraph,
    max_words: int,
) -> list[Paragraph]:
    """Split oversized paragraphs into smaller pieces."""
    words = paragraph.text.split()

    if len(words) <= max_words:
        return [paragraph]

    pieces = []

    for start in range(0, len(words), max_words):
        piece = " ".join(words[start:start + max_words])

        pieces.append(
            Paragraph(
                text=piece,
                section=paragraph.section,
                page=paragraph.page,
            )
        )

    return pieces


def chunk_paper(
    paper: ParsedPaper,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[Chunk]:
    """Convert parsed paragraphs into section-aware chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1"
        )

    chunks: list[Chunk] = []

    buffer: list[str] = []
    current_section: str | None = None
    current_page: int | None = None
    new_words = 0

    def flush(keep_overlap: bool = True) -> None:
        nonlocal buffer, current_page, new_words

        if buffer and new_words > 0:
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.paper_id}-{len(chunks)}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    section=current_section or "Unknown",
                    page=current_page or 1,
                    text=" ".join(buffer),
                )
            )

        if keep_overlap and overlap > 0:
            buffer = buffer[-overlap:]
        else:
            buffer = []

        new_words = 0
        current_page = None

    for paragraph in paper.paragraphs:
        if paragraph.section != current_section:
            if buffer:
                flush(keep_overlap=False)

            current_section = paragraph.section

        pieces = split_long_paragraph(
            paragraph,
            chunk_size - overlap,
        )

        for piece in pieces:
            words = piece.text.split()

            if not words:
                continue

            if buffer and len(buffer) + len(words) > chunk_size:
                flush(keep_overlap=True)

            if current_page is None:
                current_page = piece.page

            buffer.extend(words)
            new_words += len(words)

    if buffer and new_words > 0:
        flush(keep_overlap=False)

    return chunks
