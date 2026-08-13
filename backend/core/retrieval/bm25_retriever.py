from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retrieval using BM25.

    The retriever operates over the document chunks already
    produced by the ingestion pipeline.
    """

    def __init__(
        self,
        chunks: list[dict],
    ) -> None:
        self.chunks = chunks

        if not chunks:
            self.tokenized_corpus = []
            self.bm25 = None
            return

        self.tokenized_corpus = [
            self._tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text into simple word tokens."""

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the highest-scoring chunks for a query.
        """

        if not query.strip() or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:limit]:
            chunk = dict(self.chunks[index])

            chunk["score"] = float(scores[index])
            chunk["retrieval_method"] = "bm25"

            results.append(chunk)

        return results
