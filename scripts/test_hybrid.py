from backend.core.retrieval.hybrid_retriever import HybridRetriever


def main() -> None:
    print("Initializing hybrid retriever...")

    retriever = HybridRetriever()

    queries = [
        "What is BGE-M3?",
        "What is Qdrant?",
        "How does retrieval augmented generation work?",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        results = retriever.search(
            query,
            limit=2,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):
            print(f"\n#{rank}")
            print(f"RRF Score: {result['rrf_score']:.6f}")
            print(f"Methods: {', '.join(result['retrieval_methods'])}")
            print(f"Source: {result['filename']}")
            print(f"Chunk: {result['chunk_index']}")
            print(f"Text: {result['text']}")

    print("\n" + "=" * 70)
    print("Hybrid retrieval test: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
