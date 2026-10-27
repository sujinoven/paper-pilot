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

    # Both initial generation and repair return invalid citations.
    mock_client.chat.return_value = mock_response

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    captured = capsys.readouterr()

    assert "[9]" in answer
    assert "Citation check failed. Attempting repair." in captured.out
    assert "Citation repair failed." in captured.out
    assert "Invalid citations: [9]" in captured.out

    # Initial generation + one repair attempt.
    assert mock_client.chat.call_count == 2

@patch("paperpilot.generation.Client")
def test_generator_warns_when_citations_are_missing(
    mock_client_class,
    capsys,
):
    mock_client = mock_client_class.return_value

    mock_response = MagicMock()
    mock_response.message.content = (
        "The Transformer uses sinusoidal positional encodings."
    )

    # Both generation and repair return an uncited answer.
    mock_client.chat.return_value = mock_response

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    captured = capsys.readouterr()

    assert "sinusoidal positional encodings" in answer
    assert "Citation check failed. Attempting repair." in captured.out
    assert "Citation repair failed." in captured.out

    assert mock_client.chat.call_count == 2

@patch("paperpilot.generation.Client")
def test_generator_repairs_missing_citations(mock_client_class):
    mock_client = mock_client_class.return_value

    first_response = MagicMock()
    first_response.message.content = (
        "The Transformer uses sinusoidal positional encodings."
    )

    repaired_response = MagicMock()
    repaired_response.message.content = (
        "The Transformer uses sinusoidal positional encodings [1]."
    )

    mock_client.chat.side_effect = [
        first_response,
        repaired_response,
    ]

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    assert answer.endswith("[1].")
    assert mock_client.chat.call_count == 2

@patch("paperpilot.generation.Client")
def test_generator_keeps_original_when_repair_fails(
    mock_client_class,
    capsys,
):
    mock_client = mock_client_class.return_value

    first_response = MagicMock()
    first_response.message.content = (
        "The Transformer uses sinusoidal encodings [9]."
    )

    failed_repair = MagicMock()
    failed_repair.message.content = (
        "The Transformer uses sinusoidal encodings."
    )

    mock_client.chat.side_effect = [
        first_response,
        failed_repair,
    ]

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    captured = capsys.readouterr()

    assert answer == first_response.message.content
    assert "Citation repair failed." in captured.out
    assert mock_client.chat.call_count == 2


@patch("paperpilot.generation.Client")
def test_generator_skips_repair_for_valid_citations(
    mock_client_class,
):
    mock_client = mock_client_class.return_value

    response = MagicMock()
    response.message.content = (
        "The Transformer uses sinusoidal encodings [1]."
    )

    mock_client.chat.return_value = response

    generator = PaperAnswerGenerator()

    answer = generator.generate(
        "What positional encodings are used?",
        sample_results(),
    )

    assert answer == response.message.content
    assert mock_client.chat.call_count == 1