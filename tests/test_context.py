from paperpilot.context import format_context


def test_format_context():
    results = [
        {
            "chunk_id": "paper-17",
            "text": "The model uses sine and cosine functions.",
            "metadata": {
                "title": "Attention Is All You Need",
                "section": "3.5 Positional Encoding",
                "page": 6,
            },
        }
    ]

    context = format_context(results)

    assert "[1]" in context
    assert "Attention Is All You Need" in context
    assert "3.5 Positional Encoding" in context
    assert "Page: 6" in context
    assert "paper-17" in context
    assert "sine and cosine" in context


def test_format_multiple_passages():
    results = [
        {
            "chunk_id": "paper-1",
            "text": "First passage.",
            "metadata": {
                "title": "Paper",
                "section": "Introduction",
                "page": 1,
            },
        },
        {
            "chunk_id": "paper-2",
            "text": "Second passage.",
            "metadata": {
                "title": "Paper",
                "section": "Methods",
                "page": 2,
            },
        },
    ]

    context = format_context(results)

    assert "[1]" in context
    assert "[2]" in context
    assert "First passage." in context
    assert "Second passage." in context


def test_empty_context():
    assert format_context([]) == ""