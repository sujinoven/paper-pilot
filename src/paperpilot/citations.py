import re


def extract_citations(answer: str) -> list[int]:
    """
    Extract numbered citations such as [1], [2], [3].
    """
    matches = re.findall(r"\[(\d+)\]", answer)

    return [int(number) for number in matches]


def validate_citations(
    answer: str,
    source_count: int,
) -> dict:
    """
    Check whether all numbered citations refer
    to available retrieved passages.
    """
    if source_count < 0:
        raise ValueError("source_count cannot be negative.")

    citations = extract_citations(answer)

    valid = sorted({
        citation
        for citation in citations
        if 1 <= citation <= source_count
    })

    invalid = sorted({
        citation
        for citation in citations
        if citation < 1 or citation > source_count
    })

    return {
        "valid": valid,
        "invalid": invalid,
        "has_citations": bool(citations),
        "all_valid": len(invalid) == 0,
    }

def check_citation_coverage(
    answer: str,
    source_count: int,
) -> dict:
    """
    Check whether a generated answer includes at least one
    valid source citation, unless it is a standard refusal.
    """
    if source_count < 0:
        raise ValueError("source_count cannot be negative.")

    validation = validate_citations(
        answer=answer,
        source_count=source_count,
    )

    refusal_text = (
        "I could not find enough information in the "
        "provided paper passages to answer this question."
    )

    is_refusal = answer.strip() == refusal_text

    has_valid_citation = bool(validation["valid"])

    return {
        "is_refusal": is_refusal,
        "has_valid_citation": has_valid_citation,
        "coverage_ok": (
            is_refusal
            or (
                has_valid_citation
                and validation["all_valid"]
            )
        ),
    }