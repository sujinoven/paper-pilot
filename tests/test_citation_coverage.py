from paperpilot.citations import check_citation_coverage


def test_answer_with_valid_citation():
    result = check_citation_coverage(
        "The model uses sinusoidal encodings [1].",
        source_count=3,
    )

    assert result["coverage_ok"] is True


def test_answer_without_citation():
    result = check_citation_coverage(
        "The model uses sinusoidal encodings.",
        source_count=3,
    )

    assert result["coverage_ok"] is False


def test_answer_with_invalid_citation():
    result = check_citation_coverage(
        "The model uses sinusoidal encodings [9].",
        source_count=3,
    )

    assert result["coverage_ok"] is False


def test_refusal_does_not_require_citation():
    result = check_citation_coverage(
        "I could not find enough information in the "
        "provided paper passages to answer this question.",
        source_count=3,
    )

    assert result["coverage_ok"] is True