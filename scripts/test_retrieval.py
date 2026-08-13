from backend.core.retrieval.retriever import DenseRetriever


def main() -> None:
    print("Initializing dense retriever...")

    retriever = DenseRetriever(
        collection_name="rag_docs",
        top_k=3,
    )

    queries = [
        "What is BGE-M3?",
        "What is Qdrant?",
        "How does retrieval augmented generation work?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        results = retriever.retrieve(query)

        for rank, result in enumerate(results, start=1):
            print(f"\n#{rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['filename']}")
            print(f"Chunk: {result['chunk_index']}")
            print(f"Text: {result['text']}")

    print("\nDense retrieval test: PASSED")


if __name__ == "__main__":
    main()
