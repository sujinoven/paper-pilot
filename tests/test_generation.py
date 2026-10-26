from unittest.mock import MagicMock, patch

import pytest

from paperpilot.generation import PaperAnswerGenerator


def sample_results():
    return [
        {
            "chunk_id": "paper-001",
            "text": (
                "Multi-head attention allows the model to jointly "
                "attend to information from different representation "
                "subspaces at different positions."
            ),
            "metadata": {
                "title": "Attention Is All You Need",
                "section": "3.2.2 Multi-Head Attention",
                "page": 4,
            },
        }
    ]


@patch("paperpilot.generation.Client")
def test_generator_returns_grounded_answer(mock_client_class):
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.message.content = (
        "Multi-head attention uses different "
        "representation subspaces [1]."
    )

    mock_client.chat.return_value = mock_response

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "Why use multi-head attention?",
        sample_results(),
    )

    assert "representation subspaces" in answer
    assert "[1]" in answer
    mock_client.chat.assert_called_once()


@patch("paperpilot.generation.Client")
def test_generator_rejects_empty_question(mock_client_class):
    generator = PaperAnswerGenerator()

    with pytest.raises(ValueError):
        generator.generate("   ", sample_results())

    mock_client_class.return_value.chat.assert_not_called()


@patch("paperpilot.generation.Client")
def test_generator_handles_empty_results(mock_client_class):
    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What is the electricity cost?",
        [],
    )

    assert "could not find enough information" in answer
    mock_client_class.return_value.chat.assert_not_called()


@patch("paperpilot.generation.Client")
def test_generator_warns_about_invalid_citation(
    mock_client_class,
    capsys,
):
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.message.content = (
        "The paper uses sinusoidal encodings [9]."
    )

    mock_client.chat.return_value = mock_response

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    captured = capsys.readouterr()

    assert "[9]" in answer
    assert "Invalid citations detected" in captured.out
    assert "[9]" in captured.out