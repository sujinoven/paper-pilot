
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Paragraph:
    text: str
    page: int
    section: str


@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    paragraphs: list[Paragraph]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalize PDF text.

    - Rejoin words split by a hyphen and line break.
    - Replace repeated whitespace with a single space.

    Example:
        "The Trans-\\nformer uses attention."
        -> "The Transformer uses attention."
    """
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])",
        "",
        text,
    )

    return " ".join(text.split())


# ============================================================
# PAPER IDENTIFICATION
# ============================================================

def generate_paper_id(pdf_path: str | Path) -> str:
    """
    Generate a stable 16-character ID using SHA-256
    of the PDF file contents.
    """
    pdf_path = Path(pdf_path)

    digest = hashlib.sha256()

    with pdf_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()[:16]


def compute_paper_id(pdf_path: str | Path) -> str:
    """Alternative name for generate_paper_id()."""
    return generate_paper_id(pdf_path)


# ============================================================
# SECTION HEADING DETECTION
# ============================================================

NUMBERED_HEADING_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*\.?)\s+([A-Za-z][A-Za-z\s\-]*)$"
)

COMMON_HEADINGS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "conclusion": "Conclusion",
    "conclusions": "Conclusions",
    "references": "References",
}


def detect_section_heading(text: str) -> str | None:
    """
    Detect a numbered heading and preserve its number.

    Examples:
        "3 Model Architecture"
        "3.1 Encoder and Decoder Stacks"
        "3.2.1 Scaled Dot-Product Attention"
        "3.5 Positional Encoding"
        "4 Why Self-Attention"

    Returns None for ordinary text.
    """
    cleaned = " ".join(text.split())

    if not cleaned or len(cleaned) > 100:
        return None

    match = NUMBERED_HEADING_PATTERN.fullmatch(cleaned)

    if match is None:
        return None

    return cleaned


def detect_heading(text: str) -> str | None:
    """
    Detect a research-paper heading.

    Backward-compatible with the original PaperPilot tests.

    Examples:
        "1. Introduction" -> "Introduction"
        "3. Methods" -> "Methods"
        "Abstract" -> "Abstract"
        "References" -> "References"
        "3.5 Positional Encoding" -> "Positional Encoding"
    """
    cleaned = " ".join(text.split())

    if not cleaned or len(cleaned) > 100:
        return None

    common = COMMON_HEADINGS.get(cleaned.lower())

    if common is not None:
        return common

    match = NUMBERED_HEADING_PATTERN.fullmatch(cleaned)

    if match is not None:
        return match.group(2).strip()

    return None


# ============================================================
# TITLE EXTRACTION
# ============================================================

def extract_title(page: pymupdf.Page) -> str:
    """
    Extract the likely paper title from the first page.

    Uses font size, page position, and metadata filtering
    to avoid selecting arXiv headers or boilerplate.
    """
    page_data = page.get_text("dict")

    candidates = []
    page_height = page.rect.height

    for block in page_data["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            spans = line.get("spans", [])

            if not spans:
                continue

            text = clean_text(
                " ".join(
                    span["text"]
                    for span in spans
                )
            )

            if not text:
                continue

            font_size = max(
                span["size"]
                for span in spans
            )

            top = line["bbox"][1]
            left = line["bbox"][0]

            # Focus on the upper half of the first page.
            if top > page_height * 0.5:
                continue

            lower = text.lower()

            # Exclude common metadata and boilerplate.
            if any(
                excluded in lower
                for excluded in (
                    "arxiv:",
                    "provided proper attribution",
                    "copyright",
                    "license",
                    "http://",
                    "https://",
                    "doi:",
                )
            ):
                continue

            # Exclude lines without alphabetic content.
            if not re.search(r"[A-Za-z]", text):
                continue

            candidates.append(
                {
                    "text": text,
                    "font_size": font_size,
                    "top": top,
                    "left": left,
                }
            )

    if not candidates:
        return "Untitled Paper"

    largest_font = max(
        item["font_size"]
        for item in candidates
    )

    title_lines = [
        item
        for item in candidates
        if item["font_size"] >= largest_font - 0.5
    ]

    title_lines.sort(
        key=lambda item: (
            item["top"],
            item["left"],
        )
    )

    title = clean_text(
        " ".join(
            item["text"]
            for item in title_lines
        )
    )

    return title or "Untitled Paper"


# ============================================================
# PDF PARSING
# ============================================================

def parse_pdf(pdf_path: str | Path) -> ParsedPaper:
    """
    Parse a PDF research paper.

    Returns:
        ParsedPaper containing:
        - paper_id
        - title
        - paragraphs with page and section metadata

    Numbered headings are preserved in section metadata.
    Parsing stops when the References section is reached.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if not pdf_path.is_file():
        raise ValueError(
            f"Path is not a file: {pdf_path}"
        )

    paper_id = generate_paper_id(pdf_path)

    paragraphs: list[Paragraph] = []

    current_section = "Front Matter"
    reached_references = False

    with pymupdf.open(pdf_path) as doc:
        if len(doc) == 0:
            raise ValueError(
                "PDF contains no pages."
            )

        title = extract_title(doc[0])

        for page_index, page in enumerate(doc):
            page_number = page_index + 1

            # sort=True helps follow reading order.
            blocks = page.get_text(
                "blocks",
                sort=True,
            )

            for block in blocks:
                # Ignore image/non-text blocks.
                if len(block) > 6 and block[6] != 0:
                    continue

                raw_text = block[4]

                if not raw_text.strip():
                    continue

                # Keep original lines for heading detection.
                lines = [
                    line.strip()
                    for line in raw_text.splitlines()
                    if line.strip()
                ]

                if not lines:
                    continue

                # Ignore standalone page numbers.
                if len(lines) == 1 and lines[0].isdigit():
                    continue

                text = clean_text(raw_text)

                # ------------------------------------------------
                # Case 1: Entire block is a heading.
                # ------------------------------------------------

                numbered_heading = detect_section_heading(text)

                if numbered_heading is not None:
                    current_section = numbered_heading
                    continue

                ordinary_heading = detect_heading(text)

                if ordinary_heading is not None:
                    if ordinary_heading == "References":
                        reached_references = True
                        break

                    current_section = ordinary_heading
                    continue

                # ------------------------------------------------
                # Case 2: Heading is the first line of a block.
                # ------------------------------------------------

                first_line = lines[0]

                numbered_heading = detect_section_heading(
                    first_line
                )

                ordinary_heading = detect_heading(
                    first_line
                )

                heading = (
                    numbered_heading
                    if numbered_heading is not None
                    else ordinary_heading
                )

                if heading is not None:
                    if heading == "References":
                        reached_references = True
                        break

                    current_section = heading

                    remaining_text = clean_text(
                        "\n".join(lines[1:])
                    )

                    if not remaining_text:
                        continue

                    text = remaining_text

                # ------------------------------------------------
                # Store paragraph.
                # ------------------------------------------------

                if text:
                    paragraphs.append(
                        Paragraph(
                            text=text,
                            page=page_number,
                            section=current_section,
                        )
                    )

            if reached_references:
                break

    return ParsedPaper(
        paper_id=paper_id,
        title=title,
        paragraphs=paragraphs,
    )
