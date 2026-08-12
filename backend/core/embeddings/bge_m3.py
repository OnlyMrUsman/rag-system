from __future__ import annotations

from typing import Sequence

import torch
from sentence_transformers import SentenceTransformer


class BGEEmbeddingModel:
    """
    BGE-M3 multilingual embedding service.

    Produces 1024-dimensional dense embeddings suitable for
    semantic retrieval in Qdrant.
    """

    MODEL_NAME = "BAAI/bge-m3"
    DIMENSION = 1024

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
    ) -> list[list[float]]:
        """
        Encode a batch of texts into dense vectors.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def encode_one(self, text: str) -> list[float]:
        """
        Encode a single text.
        """

        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def device_name(self) -> str:
        return self.device
