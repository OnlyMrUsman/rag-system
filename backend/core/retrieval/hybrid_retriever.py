from __future__ import annotations

from typing import Any

from backend.core.ingestion.document_loader import DocumentLoader
from backend.core.ingestion.chunker import TextChunker
from backend.core.retrieval.bm25_retriever import BM25Retriever
from backend.core.retrieval.retriever import DenseRetriever


class HybridRetriever:
    """
    Hybrid retrieval combining dense semantic search and BM25
    keyword search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        collection_name: str = "rag_docs",
        dense_limit: int = 10,
        bm25_limit: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.dense_limit = dense_limit
        self.bm25_limit = bm25_limit
        self.rrf_k = rrf_k

        # Dense semantic retriever.
        self.dense_retriever = DenseRetriever(
            collection_name=collection_name,
            top_k=dense_limit,
        )

        # Build the lightweight BM25 index without loading
        # another embedding model.
        loader = DocumentLoader(
            documents_dir=documents_dir,
        )

        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks = []

        for document in loader.load_all():
            chunks.extend(
                chunker.chunk_document(document)
            )

        self.bm25_retriever = BM25Retriever(
            chunks
        )

    @staticmethod
    def _chunk_key(
        result: dict[str, Any],
    ) -> tuple[str, int]:
        """Create a stable identifier for a document chunk."""

        return (
            result["source"],
            int(result["chunk_index"]),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve using dense + BM25 and combine rankings
        using Reciprocal Rank Fusion.

        RRF:
            score = sum(1 / (k + rank))
        """

        if not query.strip():
            return []

        dense_results = self.dense_retriever.retrieve(
            query,
            top_k=self.dense_limit,
        )

        bm25_results = self.bm25_retriever.search(
            query,
            limit=self.bm25_limit,
        )

        fused: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}

        # -------------------------------------------------
        # Dense ranking
        # -------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            key = self._chunk_key(result)

            if key not in fused:
                fused[key] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_methods": set(),
                }

            fused[key]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )

            fused[key]["retrieval_methods"].add(
                "dense"
            )

        # -------------------------------------------------
        # BM25 ranking
        # -------------------------------------------------

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            key = self._chunk_key(result)

            if key not in fused:
                fused[key] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_methods": set(),
                }

            fused[key]["rrf_score"] += (
                1.0 / (self.rrf_k + rank)
            )

            fused[key]["retrieval_methods"].add(
                "bm25"
            )

        # -------------------------------------------------
        # Final fused ranking
        # -------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        final_results = []

        for result in ranked[:limit]:
            result = dict(result)

            result["retrieval_methods"] = sorted(
                result["retrieval_methods"]
            )

            result["retrieval_method"] = "hybrid"

            final_results.append(result)

        return final_results
