from paperpilot.vector_store import PaperVectorStore
from paperpilot.retrieval import retrieve
from paperpilot.generation import PaperAnswerGenerator


def main():
    store = PaperVectorStore()
    generator = PaperAnswerGenerator()

    questions = [
        "What positional encodings are used in the Transformer?",
        "Why does the Transformer use multi-head attention?",
        "What was the exact electricity bill paid by the authors while training the Transformer?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        results = retrieve(
            store,
            question,
            top_k=3,
            candidate_k=15,
        )

        print("\nRETRIEVED SOURCES")

        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(
                f"[{index}] "
                f"{metadata['section']} | "
                f"Page {metadata['page']}"
            )

        answer = generator.generate(question, results)

        print("\nGENERATED ANSWER")
        print(answer)


if __name__ == "__main__":
    main()