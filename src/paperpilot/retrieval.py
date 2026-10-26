
import re

from .vector_store import PaperVectorStore


STOPWORDS = {
    "what", "which", "where", "when", "why", "how",
    "the", "a", "an", "is", "are", "was", "were",
    "in", "on", "of", "for", "to", "and", "does",
    "do", "used", "use",
}


def tokenize(text: str) -> set[str]:
    """Extract meaningful lowercase words."""
    words = re.findall(r"[a-zA-Z]+", text.lower())

    return {
        word
        for word in words
        if word not in STOPWORDS and len(word) > 2
    }


def keyword_overlap(question: str, passage: str) -> float:
    """Calculate fraction of query terms found in a passage."""
    query_terms = tokenize(question)
    passage_terms = tokenize(passage)

    if not query_terms:
        return 0.0

    return len(query_terms & passage_terms) / len(query_terms)

def section_relevance(question: str, section: str) -> float:
    """
    Measure overlap between the question and section heading.
    """
    question_tokens = set(tokenize(question))
    section_tokens = set(tokenize(section))

    if not question_tokens or not section_tokens:
        return 0.0

    return len(question_tokens & section_tokens) / len(question_tokens)


def retrieve(
    store,
    question: str,
    top_k: int = 5,
    candidate_k: int = 15,
) -> list[dict]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be greater than or equal to top_k."
        )

    candidates = store.search(
        question,
        top_k=candidate_k,
    )

    reranked = []

    for result in candidates:
        semantic_score = 1.0 - result["distance"]

        keyword_score = keyword_overlap(
            question,
            result["text"],
        )

        section_score = section_relevance(
            question,
            result["metadata"].get("section", ""),
        )

        combined_score = (
            0.65 * semantic_score
            + 0.15 * keyword_score
            + 0.20 * section_score
        )

        reranked.append(
            {
                **result,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "section_score": section_score,
                "combined_score": combined_score,
            }
        )

    reranked.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return reranked[:top_k]