from paperpilot.evaluation import EvaluationCase, evaluate_answer


def test_flags_known_misleading_phrase():
    case = EvaluationCase(
        question="Why use residual connections?",
        expected_section="3.1 Encoder and Decoder Stacks",
        answerable=True,
        forbidden_phrases=["bypassing the normalization layer"],
    )

    result = evaluate_answer(
        case,
        "Residual connections help by bypassing the normalization layer [1].",
        source_count=3,
    )

    assert result["has_quality_flags"] is True
    assert result["flagged_phrases"] == [
        "bypassing the normalization layer"
    ]


def test_does_not_flag_clean_answer():
    case = EvaluationCase(
        question="Why use residual connections?",
        expected_section="3.1 Encoder and Decoder Stacks",
        answerable=True,
        forbidden_phrases=["bypassing the normalization layer"],
    )

    result = evaluate_answer(
        case,
        "Residual connections add the input to the sublayer output [1].",
        source_count=3,
    )

    assert result["has_quality_flags"] is False
    assert result["flagged_phrases"] == []