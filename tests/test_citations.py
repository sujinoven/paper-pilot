import pytest

from paperpilot.citations import (
    extract_citations,
    validate_citations,
)


def test_extract_citations():
    answer = "Multi-head attention uses multiple heads [1] and [2]."

    assert extract_citations(answer) == [1, 2]


def test_validate_correct_citations():
    result = validate_citations(
        "The Transformer uses sinusoidal encodings [1].",
        source_count=3,
    )

    assert result["valid"] == [1]
    assert result["invalid"] == []
    assert result["all_valid"] is True


def test_detect_invalid_citation():
    result = validate_citations(
        "Learned embeddings were also tested [9].",
        source_count=3,
    )

    assert result["invalid"] == [9]
    assert result["all_valid"] is False


def test_answer_without_citations():
    result = validate_citations(
        "There is not enough information.",
        source_count=3,
    )

    assert result["has_citations"] is False
    assert result["all_valid"] is True


def test_reject_negative_source_count():
    with pytest.raises(ValueError):
        validate_citations("Answer [1]", source_count=-1)