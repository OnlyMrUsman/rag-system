from backend.core.ingestion.indexer import DocumentIndexer


def main() -> None:
    print("Initializing document indexer...")

    indexer = DocumentIndexer()

    print(f"Documents directory: {indexer.documents_dir}")
    print(f"Embedding device: {indexer.embedder.device_name}")
    print(f"Embedding dimension: {indexer.embedder.dimension}")

    chunks = indexer.build_chunks()

    print(f"\nChunks discovered: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nChunk {chunk['chunk_index']}"
            f" | {chunk['filename']}"
        )
        print(chunk["text"])

    count = indexer.index(recreate=True)

    print(f"\nIndexed vectors: {count}")
    print(f"Qdrant vectors: {indexer.vector_store.count()}")

    assert count == len(chunks)
    assert indexer.vector_store.count() == len(chunks)

    print("\nDocument indexing test: PASSED")


if __name__ == "__main__":
    main()
