from __future__ import annotations

from typing import Any

from backend.core.embeddings.bge_m3 import BGEEmbeddingModel
from backend.core.retrieval.qdrant_store import QdrantVectorStore


class DenseRetriever:
    """
    Dense semantic retriever using BGE-M3 + Qdrant.
    """

    def __init__(
        self,
        collection_name: str = "rag_docs",
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k

        self.embedder = BGEEmbeddingModel()

        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            vector_size=self.embedder.dimension,
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically relevant chunks.
        """

        if not query.strip():
            return []

        limit = top_k or self.top_k

        query_vector = self.embedder.encode_one(query)

        results = self.vector_store.search(
            vector=query_vector,
            limit=limit,
        )

        retrieved = []

        for result in results:
            payload = result.payload or {}

            retrieved.append(
                {
                    "score": float(result.score),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "filename": payload.get("filename", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                }
            )

        return retrieved
