from paperpilot.vector_store import PaperVectorStore
from paperpilot.evaluation import evaluate_retrieval


def main():
    store = PaperVectorStore()

    report = evaluate_retrieval(
        store,
        top_k=3,
    )

    print("\nRETRIEVAL EVALUATION")
    print("=" * 60)

    for item in report["details"]:
        print(f"\nQuestion: {item['question']}")
        print(f"Expected section: {item['expected_section']}")
        print(f"Retrieved sections: {item['retrieved_sections']}")
        print(f"Hit: {item['hit']}")

    print("\n" + "=" * 60)
    print(
        f"Hit Rate@3: {report['hits']}/{report['evaluated']} "
        f"({report['hit_rate']:.1%})"
    )


if __name__ == "__main__":
    main()