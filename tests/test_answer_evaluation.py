from paperpilot.evaluation import (
    EvaluationCase,
    evaluate_answer,
)


def test_answerable_response_with_citation_and_keywords():
    case = EvaluationCase(
        question="Why use multi-head attention?",
        expected_section="3.2.2 Multi-Head Attention",
        answerable=True,
        expected_keywords=["subspaces", "positions"],
    )

    result = evaluate_answer(
        case,
        "It attends to different subspaces and positions [2].",
        source_count=3,
    )

    assert result["refusal_correct"] is True
    assert result["citation_ok"] is True
    assert result["keyword_coverage"] == 1.0


def test_answerable_response_without_citation():
    case = EvaluationCase(
        question="What encodings are used?",
        expected_section="3.5 Positional Encoding",
        answerable=True,
        expected_keywords=["sinusoidal", "learned"],
    )

    result = evaluate_answer(
        case,
        "Sinusoidal and learned positional embeddings are discussed.",
        source_count=3,
    )

    assert result["refusal_correct"] is True
    assert result["citation_ok"] is False
    assert result["keyword_coverage"] == 1.0


def test_correct_refusal():
    case = EvaluationCase(
        question="What was the electricity bill?",
        expected_section=None,
        answerable=False,
    )

    result = evaluate_answer(
        case,
        "I could not find enough information in the "
        "provided paper passages to answer this question.",
        source_count=3,
    )

    assert result["is_refusal"] is True
    assert result["refusal_correct"] is True
    assert result["citation_ok"] is None
    assert result["keyword_coverage"] is None


def test_incorrect_refusal_for_answerable_question():
    case = EvaluationCase(
        question="Why is attention scaled?",
        expected_section="3.2.1 Scaled Dot-Product Attention",
        answerable=True,
        expected_keywords=["softmax"],
    )

    result = evaluate_answer(
        case,
        "I could not find enough information in the "
        "provided paper passages to answer this question.",
        source_count=3,
    )

    assert result["refusal_correct"] is False
    assert result["citation_ok"] is True
    assert result["keyword_coverage"] == 0.0