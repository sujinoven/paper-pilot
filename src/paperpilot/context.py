import re


def remove_bibliography_citations(text: str) -> str:
    """
    Remove numeric bibliography references from paper passages.
    """
    pattern = r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]"

    cleaned = re.sub(pattern, "", text)

    return re.sub(r"[ \t]{2,}", " ", cleaned)


def remove_corrupted_positional_equations(text: str) -> str:
    """
    Remove recognizable corrupted positional-encoding equation lines.

    This is a targeted safeguard, not a general math parser.
    """
    cleaned_lines = []

    for line in text.splitlines():
        normalized = line.lower().replace(" ", "")

        is_positional_equation = (
            "pe(pos" in normalized
        )

        if is_positional_equation:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def format_context(results: list[dict]) -> str:
    """
    Format retrieved passages with numbered citations and metadata.
    """
    if not results:
        return ""

    passages = []

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]

        title = metadata.get("title", "Unknown Paper")
        section = metadata.get("section", "Unknown Section")
        page = metadata.get("page", "?")

        text = remove_bibliography_citations(result["text"])

        if "positional encoding" in section.lower():
            text = remove_corrupted_positional_equations(text)

        passage = (
            f"[{index}] {title}\n"
            f"Section: {section}\n"
            f"Page: {page}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Passage:\n{text}"
        )

        passages.append(passage)

    return "\n\n---\n\n".join(passages)