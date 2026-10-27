from unittest.mock import patch

import pytest

from paperpilot.evaluation import (
    EvaluationCase,
    evaluate_retrieval,
)


def make_result(section: str) -> dict:
    return {
        "metadata": {
            "section": section,
        }
    }


@patch("paperpilot.retrieval.retrieve")
def test_evaluation_all_hits(mock_retrieve):
    mock_retrieve.side_effect = [
        [make_result("Section A")],
        [make_result("Section B")],
    ]

    cases = [
        EvaluationCase("Question A", "Section A", True),
        EvaluationCase("Question B", "Section B", True),
    ]

    report = evaluate_retrieval(
        store=None,
        cases=cases,
        top_k=3,
    )

    assert report["evaluated"] == 2
    assert report["hits"] == 2
    assert report["hit_rate"] == 1.0


@patch("paperpilot.retrieval.retrieve")
def test_evaluation_partial_hits(mock_retrieve):
    mock_retrieve.side_effect = [
        [make_result("Section A")],
        [make_result("Wrong Section")],
    ]

    cases = [
        EvaluationCase("Question A", "Section A", True),
        EvaluationCase("Question B", "Section B", True),
    ]

    report = evaluate_retrieval(
        store=None,
        cases=cases,
    )

    assert report["hits"] == 1
    assert report["evaluated"] == 2
    assert report["hit_rate"] == 0.5


@patch("paperpilot.retrieval.retrieve")
def test_unanswerable_question_excluded(mock_retrieve):
    mock_retrieve.return_value = [
        make_result("Unrelated Section")
    ]

    cases = [
        EvaluationCase(
            question="Unknown fact?",
            expected_section=None,
            answerable=False,
        ),
    ]

    report = evaluate_retrieval(
        store=None,
        cases=cases,
    )

    assert report["evaluated"] == 0
    assert report["hits"] == 0
    assert report["hit_rate"] == 0.0
    assert report["details"][0]["hit"] is None


def test_invalid_top_k():
    with pytest.raises(ValueError):
        evaluate_retrieval(
            store=None,
            cases=[],
            top_k=0,
        )