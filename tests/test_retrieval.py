
import pytest

from paperpilot.retrieval import (
    tokenize,
    keyword_overlap,
    retrieve,
)


def test_tokenize_removes_stopwords():
    tokens = tokenize(
        "What positional encodings are used in the Transformer?"
    )

    assert "positional" in tokens
    assert "encodings" in tokens
    assert "transformer" in tokens
    assert "what" not in tokens
    assert "the" not in tokens


def test_keyword_overlap_exact_match():
    score = keyword_overlap(
        "positional encodings",
        "The model uses positional encodings.",
    )

    assert score == 1.0


def test_keyword_overlap_no_match():
    score = keyword_overlap(
        "positional encodings",
        "The weather is sunny.",
    )

    assert score == 0.0


def test_keyword_overlap_empty_question():
    score = keyword_overlap(
        "",
        "Some passage text.",
    )

    assert score == 0.0


def test_retrieve_rejects_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve(
            store=None,
            question="test",
            top_k=0,
        )


class FakeVectorStore:
    """Simulate search results without loading a model."""

    def search(self, question, top_k):
        return [
            {
                "chunk_id": "general",
                "text": "Transformer architecture and attention.",
                "distance": 0.2,
                "metadata": {"page": 3},
            },
            {
                "chunk_id": "specific",
                "text": (
                    "The Transformer uses positional encodings "
                    "with sine and cosine functions."
                ),
                "distance": 0.3,
                "metadata": {"page": 6},
            },
        ]


def test_retrieve_reranks_relevant_passage():
    store = FakeVectorStore()

    results = retrieve(
        store,
        "What positional encodings are used in the Transformer?",
        top_k=2,
    )

    assert results[0]["chunk_id"] == "specific"
    assert results[0]["metadata"]["page"] == 6
    assert results[0]["combined_score"] > results[1]["combined_score"]
