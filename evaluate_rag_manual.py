from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve
from paperpilot.generation import PaperAnswerGenerator
from paperpilot.evaluation import (
    EVALUATION_CASES,
    evaluate_answer,
)


def main():
    print("Initializing PaperPilot evaluation...")

    # Initialize the vector database and local LLM.
    store = PaperVectorStore()
    generator = PaperAnswerGenerator(
        model="qwen2.5:3b",
    )

    # Evaluation counters.
    retrieval_hits = 0
    retrieval_total = 0

    citation_passes = 0
    citation_total = 0

    refusal_correct = 0

    keyword_scores = []

    # NEW: Count answers containing known problematic phrases.
    quality_flagged_cases = 0

    # Evaluate each question.
    for index, case in enumerate(EVALUATION_CASES, start=1):
        print("\n" + "=" * 70)
        print(f"CASE {index}: {case.question}")

        # Step 1: Retrieve relevant paper passages.
        results = retrieve(
            store,
            case.question,
            top_k=3,
            candidate_k=15,
        )

        retrieved_sections = [
            result["metadata"].get("section", "")
            for result in results
        ]

        # Step 2: Evaluate retrieval.
        retrieval_hit = None

        if case.expected_section is not None:
            retrieval_total += 1

            retrieval_hit = (
                case.expected_section in retrieved_sections
            )

            if retrieval_hit:
                retrieval_hits += 1

        # Step 3: Generate a grounded answer.
        answer = generator.generate(
            question=case.question,
            results=results,
        )

        # Step 4: Evaluate the generated answer.
        evaluation = evaluate_answer(
            case=case,
            answer=answer,
            source_count=len(results),
        )

        # NEW: Track answers with quality flags.
        if evaluation["has_quality_flags"]:
            quality_flagged_cases += 1

        # Step 5: Track refusal correctness.
        if evaluation["refusal_correct"]:
            refusal_correct += 1

        # Step 6: Track citation coverage for answerable cases.
        if case.answerable:
            citation_total += 1

            if evaluation["citation_ok"]:
                citation_passes += 1

        # Step 7: Track keyword coverage.
        if evaluation["keyword_coverage"] is not None:
            keyword_scores.append(
                evaluation["keyword_coverage"]
            )

        # Step 8: Print individual results.
        print("\nRetrieved sections:")

        for section in retrieved_sections:
            print(f"  - {section}")

        print("\nGenerated answer:")
        print(answer)

        print("\nEvaluation:")
        print("  Retrieval hit:", retrieval_hit)
        print("  Refusal correct:", evaluation["refusal_correct"])
        print("  Citation OK:", evaluation["citation_ok"])
        print("  Keyword coverage:", evaluation["keyword_coverage"])

        # NEW: Display detected quality issues.
        print("  Quality flags:", evaluation["flagged_phrases"])

    # Step 9: Calculate aggregate metrics.
    retrieval_rate = (
        retrieval_hits / retrieval_total
        if retrieval_total
        else 0.0
    )

    citation_rate = (
        citation_passes / citation_total
        if citation_total
        else 0.0
    )

    average_keyword_coverage = (
        sum(keyword_scores) / len(keyword_scores)
        if keyword_scores
        else 0.0
    )

    # Step 10: Print final evaluation summary.
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Retrieval Hit Rate@3: "
        f"{retrieval_hits}/{retrieval_total} "
        f"({retrieval_rate:.1%})"
    )

    print(
        f"Citation Coverage Pass Rate: "
        f"{citation_passes}/{citation_total} "
        f"({citation_rate:.1%})"
    )

    print(
        f"Refusal Correctness: "
        f"{refusal_correct}/{len(EVALUATION_CASES)}"
    )

    print(
        f"Average Keyword Coverage: "
        f"{average_keyword_coverage:.1%}"
    )

    # NEW: Summary of factual-quality flags.
    print(
        f"Answers flagged for review: "
        f"{quality_flagged_cases}/{len(EVALUATION_CASES)}"
    )


if __name__ == "__main__":
    main()