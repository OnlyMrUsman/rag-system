from uuid import uuid4

from qdrant_client.models import PointStruct

from backend.core.embeddings.bge_m3 import BGEEmbeddingModel
from backend.core.retrieval.qdrant_store import QdrantVectorStore


COLLECTION_NAME = "rag_docs"


def main() -> None:
    embedder = BGEEmbeddingModel()

    store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        vector_size=embedder.dimension,
    )

    store.create_collection(recreate=True)

    documents = [
        {
            "text": "Qdrant is a vector database designed for similarity search.",
            "source": "qdrant_intro.txt",
        },
        {
            "text": "BGE-M3 is a multilingual embedding model supporting many languages.",
            "source": "bge_intro.txt",
        },
        {
            "text": "Retrieval augmented generation combines retrieval with language generation.",
            "source": "rag_intro.txt",
        },
    ]

    texts = [doc["text"] for doc in documents]

    vectors = embedder.encode(texts)

    points = [
        PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "text": document["text"],
                "source": document["source"],
            },
        )
        for document, vector in zip(documents, vectors)
    ]

    store.upsert(points)

    print(f"Indexed vectors: {store.count()}")

    query = "How does vector search work?"

    query_vector = embedder.encode_one(query)

    results = store.search(
        query_vector,
        limit=3,
    )

    print("\nSearch results:")

    for index, result in enumerate(results, start=1):
        print(f"\n#{index}")
        print(f"Score: {result.score:.4f}")
        print(f"Source: {result.payload['source']}")
        print(f"Text: {result.payload['text']}")

    print("\nQdrant integration test: PASSED")


if __name__ == "__main__":
    main()
