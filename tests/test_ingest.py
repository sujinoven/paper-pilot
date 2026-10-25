
import pytest

from paperpilot.ingest import (
    clean_text,
    detect_heading,
    generate_paper_id,
    parse_pdf,
)


def test_clean_text():
    raw = "The Trans-\nformer   uses\nattention."
    expected = "The Transformer uses attention."

    assert clean_text(raw) == expected


def test_detect_heading():
    assert detect_heading("1. Introduction") == "Introduction"
    assert detect_heading("3. Methods") == "Methods"
    assert detect_heading("Abstract") == "Abstract"
    assert detect_heading("References") == "References"
    assert detect_heading("This is a normal sentence.") is None


def test_hash_is_deterministic(tmp_path):
    file = tmp_path / "sample.pdf"
    file.write_bytes(b"sample pdf contents")

    first = generate_paper_id(file)
    second = generate_paper_id(file)

    assert first == second
    assert len(first) == 16


def test_missing_pdf(tmp_path):
    missing = tmp_path / "nonexistent.pdf"

    with pytest.raises(FileNotFoundError):
        parse_pdf(missing)


from pathlib import Path

import pytest

from paperpilot.ingest import parse_pdf


def test_real_paper_title():
    path = Path(
        "data/papers/attention_is_all_you_need.pdf"
    )

    if not path.exists():
        pytest.skip("Sample research paper not downloaded")

    paper = parse_pdf(path)

    assert paper.title == "Attention Is All You Need"
