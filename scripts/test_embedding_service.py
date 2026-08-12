from backend.core.embeddings.bge_m3 import BGEEmbeddingModel


def main() -> None:
    embedder = BGEEmbeddingModel()

    print(f"Device: {embedder.device_name}")
    print(f"Dimension: {embedder.dimension}")

    texts = [
        "Qdrant is a vector database.",
        "Qdrant 是一个向量数据库。",
        "Artificial intelligence enables intelligent information retrieval.",
    ]

    embeddings = embedder.encode(texts)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    print("\nEmbedding service test: PASSED")


if __name__ == "__main__":
    main()
