from backend.core.ingestion.indexer import DocumentIndexer
from backend.core.retrieval.bm25_retriever import BM25Retriever


def main() -> None:
    print("Building BM25 index...")

    indexer = DocumentIndexer()

    chunks = indexer.build_chunks()

    print(f"Chunks indexed for BM25: {len(chunks)}")

    retriever = BM25Retriever(chunks)

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

        for rank, result in enumerate(results, start=1):
            print(f"\n#{rank}")
            print(f"Score: {result['score']:.4f}")
            print(f"Source: {result['filename']}")
            print(f"Chunk: {result['chunk_index']}")
            print(f"Text: {result['text']}")

    print("\n" + "=" * 70)
    print("BM25 retrieval test: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
