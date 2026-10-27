from dataclasses import dataclass, field
from paperpilot.citations import check_citation_coverage



@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_section: str | None
    answerable: bool
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_phrases: list[str] = field(default_factory=list)


EVALUATION_CASES = [
    EvaluationCase(
        question="What positional encodings are used in the Transformer?",
        expected_section="3.5 Positional Encoding",
        answerable=True,
        expected_keywords=["sinusoidal", "learned"],
    ),
    EvaluationCase(
        question="Why does the Transformer use multi-head attention?",
        expected_section="3.2.2 Multi-Head Attention",
        answerable=True,
        expected_keywords=["subspaces", "positions"],
    ),
    EvaluationCase(
        question="Why is dot-product attention scaled?",
        expected_section="3.2.1 Scaled Dot-Product Attention",
        answerable=True,
        expected_keywords=["softmax", "gradients"],
    ),
    EvaluationCase(
        question="What is the purpose of positional encoding?",
        expected_section="3.5 Positional Encoding",
        answerable=True,
        expected_keywords=["position"],
    ),
    EvaluationCase(
        question="What is the purpose of the feed-forward networks?",
        expected_section="3.3 Position-wise Feed-Forward Networks",
        answerable=True,
        expected_keywords=["linear", "relu"],
    ),
    EvaluationCase(
    question="Why does the Transformer use residual connections?",
    expected_section="3.1 Encoder and Decoder Stacks",
    answerable=True,
    expected_keywords=["residual", "normalization"],
    forbidden_phrases=[
        "bypassing the normalization layer",
        "bypass layer normalization",
        "facilitate these residual connections",
    ],
),

    EvaluationCase(
        question="What optimizer was used to train the Transformer?",
        expected_section="5.3 Optimizer",
        answerable=True,
        expected_keywords=["adam"],
    ),
    EvaluationCase(
        question="What is label smoothing in the Transformer training setup?",
        expected_section="5.4 Regularization",
        answerable=True,
        expected_keywords=["smoothing"],
        forbidden_phrases=[
    "reducing overfitting",
],
    ),
    EvaluationCase(
        question="How does the Transformer handle encoder-decoder attention?",
        expected_section="3.2.3 Applications of Attention in our Model",
        answerable=True,
        expected_keywords=["queries", "keys", "values"],
    ),
    EvaluationCase(
        question="What was the exact electricity bill paid by the authors while training the Transformer?",
        expected_section=None,
        answerable=False,
    ),
    EvaluationCase(
        question="What was the home address of the first author?",
        expected_section=None,
        answerable=False,
    ),
    EvaluationCase(
        question="What brand of laptop did the authors use to write the paper?",
        expected_section=None,
        answerable=False,
    ),
]

def evaluate_retrieval(
    store,
    cases: list[EvaluationCase] = EVALUATION_CASES,
    top_k: int = 3,
) -> dict:
    """
    Evaluate whether the expected paper section appears
    among the retrieved passages.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    from paperpilot.retrieval import retrieve

    details = []
    evaluated = 0
    hits = 0

    for case in cases:
        results = retrieve(
            store,
            case.question,
            top_k=top_k,
            candidate_k=max(15, top_k),
        )

        retrieved_sections = [
            result["metadata"].get("section", "")
            for result in results
        ]

        hit = None

        if case.expected_section is not None:
            evaluated += 1
            hit = case.expected_section in retrieved_sections

            if hit:
                hits += 1

        details.append(
            {
                "question": case.question,
                "expected_section": case.expected_section,
                "retrieved_sections": retrieved_sections,
                "hit": hit,
            }
        )

    return {
        "evaluated": evaluated,
        "hits": hits,
        "hit_rate": hits / evaluated if evaluated else 0.0,
        "details": details,
    }

def evaluate_answer(
    case: EvaluationCase,
    answer: str,
    source_count: int,
) -> dict:
    """
    Evaluate basic properties of one generated answer.

    This does not establish factual correctness or faithfulness.
    """
    coverage = check_citation_coverage(
        answer=answer,
        source_count=source_count,
    )

    is_refusal = coverage["is_refusal"]

    refusal_correct = (
        not is_refusal if case.answerable else is_refusal
    )

    answer_lower = answer.lower()
    flagged_phrases = [
    phrase
    for phrase in case.forbidden_phrases
    if phrase.lower() in answer_lower
]

    matched_keywords = [
        keyword
        for keyword in case.expected_keywords
        if keyword.lower() in answer_lower
    ]

    total_keywords = len(case.expected_keywords)

    keyword_coverage = (
        len(matched_keywords) / total_keywords
        if total_keywords
        else None
    )

    citation_ok = (
        coverage["coverage_ok"]
        if case.answerable
        else None
    )

    return {
        "question": case.question,
        "answerable": case.answerable,
        "is_refusal": is_refusal,
        "refusal_correct": refusal_correct,
        "citation_ok": citation_ok,
        "keyword_coverage": keyword_coverage,
        "matched_keywords": matched_keywords,
        "flagged_phrases": flagged_phrases,
"has_quality_flags": bool(flagged_phrases),
    }