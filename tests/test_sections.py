
import pytest

from paperpilot.ingest import detect_section_heading


@pytest.mark.parametrize(
    "text",
    [
        "3 Model Architecture",
        "3.1 Encoder and Decoder Stacks",
        "3.2 Attention",
        "3.2.1 Scaled Dot-Product Attention",
        "3.5 Positional Encoding",
        "4 Why Self-Attention",
        "5.3 Optimizer",
    ],
)
def test_detect_valid_section_headings(text):
    assert detect_section_heading(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Figure 1: The Transformer - model architecture.",
        "The Transformer uses multi-head attention.",
        "3",
        "5",
        "PE(pos,2i) = sin(pos/100002i/dmodel)",
        "FFN(x) = max(0, xW1 + b1)W2 + b2 (2)",
        "",
    ],
)
def test_reject_non_headings(text):
    assert detect_section_heading(text) is None
