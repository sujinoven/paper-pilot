from paperpilot.context import remove_corrupted_positional_equations


def test_remove_positional_equation():
    text = (
        "The model uses sinusoidal encodings.\n"
        "PE(pos,2i) = sin(pos/100002i/dmodel)\n"
        "These encodings represent token positions."
    )

    result = remove_corrupted_positional_equations(text)

    assert "PE(pos,2i)" not in result
    assert "sinusoidal encodings" in result
    assert "token positions" in result


def test_preserve_normal_text():
    text = "The model uses sine and cosine functions."

    assert remove_corrupted_positional_equations(text) == text


def test_preserve_other_equations():
    text = "Attention(Q,K,V) = softmax(QK^T/sqrt(dk))V"

    assert remove_corrupted_positional_equations(text) == text