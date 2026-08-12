from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


class QdrantVectorStore:
    """
    Qdrant vector database abstraction for the RAG system.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "rag_docs",
        vector_size: int = 1024,
    ) -> None:
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        self.vector_size = vector_size

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the RAG document collection.
        """

        collections = self.client.get_collections().collections

        exists = any(
            collection.name == self.collection_name
            for collection in collections
        )

        if exists and recreate:
            self.client.delete_collection(
                collection_name=self.collection_name
            )
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        points: list[PointStruct],
    ) -> None:
        """
        Insert or update vectors in Qdrant.
        """

        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ) -> list[Any]:
        """
        Perform dense vector similarity search.
        """

        return self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        ).points

    def count(self) -> int:
        """
        Return the number of indexed vectors.
        """

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count
