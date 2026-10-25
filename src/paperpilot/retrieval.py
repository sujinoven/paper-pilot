
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


def retrieve(
    store: PaperVectorStore,
    question: str,
    top_k: int = 5,
    candidate_k: int = 15,
) -> list[dict]:
    """Retrieve candidates and rerank using keyword overlap."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    candidates = store.search(
        question,
        top_k=max(top_k, candidate_k),
    )

    reranked = []

    for result in candidates:
        semantic_score = 1.0 - result["distance"]

        keyword_score = keyword_overlap(
            question,
            result["text"],
        )

        combined_score = (
            0.7 * semantic_score
            + 0.3 * keyword_score
        )

        reranked.append(
            {
                **result,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
            }
        )

    reranked.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    return reranked[:top_k]
