from paperpilot.context import remove_bibliography_citations


def test_remove_single_bibliography_reference():
    text = "Learned embeddings [9] gave similar results."

    assert remove_bibliography_citations(text) == (
        "Learned embeddings gave similar results."
    )


def test_remove_multiple_references():
    text = "Previous methods [12, 15] used recurrence."

    assert remove_bibliography_citations(text) == (
        "Previous methods used recurrence."
    )


def test_remove_reference_range():
    text = "Related approaches [3-5] were investigated."

    assert remove_bibliography_citations(text) == (
        "Related approaches were investigated."
    )


def test_preserve_normal_numbers():
    text = "The model uses 8 attention heads and 512 dimensions."

    assert remove_bibliography_citations(text) == text


def test_preserve_equation_text():
    text = "PE(pos,2i) = sin(pos/10000^2i/dmodel)"

    assert remove_bibliography_citations(text) == text