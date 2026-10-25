
import pytest

from paperpilot.ingest import Paragraph, ParsedPaper
from paperpilot.chunking import chunk_paper


def make_paper(paragraphs):
    return ParsedPaper(
        paper_id="test123",
        title="Test Research Paper",
        paragraphs=paragraphs,
    )


def test_empty_paper():
    paper = make_paper([])

    assert chunk_paper(paper) == []


def test_single_paragraph():
    paper = make_paper([
        Paragraph(
            text="This is a short research paragraph.",
            section="Abstract",
            page=1,
        )
    ])

    chunks = chunk_paper(paper)

    assert len(chunks) == 1
    assert chunks[0].section == "Abstract"
    assert chunks[0].page == 1


def test_section_boundaries():
    paper = make_paper([
        Paragraph(
            text="Introduction " * 50,
            section="Introduction",
            page=1,
        ),
        Paragraph(
            text="Methods " * 50,
            section="Methods",
            page=2,
        ),
    ])

    chunks = chunk_paper(
        paper,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 2
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Methods"


def test_overlap():
    words = [f"word{i}" for i in range(20)]

    paper = make_paper([
        Paragraph(
            text=" ".join(words),
            section="Methods",
            page=1,
        )
    ])

    chunks = chunk_paper(
        paper,
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) >= 2

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()

    assert first_words[-2:] == second_words[:2]


def test_invalid_configuration():
    paper = make_paper([])

    with pytest.raises(ValueError):
        chunk_paper(paper, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_paper(paper, chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        chunk_paper(paper, chunk_size=100, overlap=-1)


def test_unique_chunk_ids():
    paper = make_paper([
        Paragraph(
            text=" ".join(f"word{i}" for i in range(100)),
            section="Methods",
            page=1,
        )
    ])

    chunks = chunk_paper(
        paper,
        chunk_size=30,
        overlap=5,
    )

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))
