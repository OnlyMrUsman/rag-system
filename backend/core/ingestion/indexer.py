from __future__ import annotations

from pathlib import Path

from qdrant_client.models import PointStruct

from backend.core.embeddings.bge_m3 import BGEEmbeddingModel
from backend.core.ingestion.chunker import TextChunker
from backend.core.ingestion.document_loader import DocumentLoader
from backend.core.retrieval.qdrant_store import QdrantVectorStore


class DocumentIndexer:
    """
    End-to-end document indexing pipeline.

    Documents
        ↓
    DocumentLoader
        ↓
    TextChunker
        ↓
    BGE-M3 embeddings
        ↓
    Qdrant
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        collection_name: str = "rag_docs",
        batch_size: int = 8,
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.batch_size = batch_size

        self.loader = DocumentLoader(
            documents_dir=str(self.documents_dir)
        )

        self.chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100,
        )

        self.embedder = BGEEmbeddingModel()

        self.vector_store = QdrantVectorStore(
            collection_name=collection_name,
            vector_size=self.embedder.dimension,
        )

    def build_chunks(self) -> list[dict]:
        """Load all documents and split them into chunks."""

        documents = self.loader.load_all()

        all_chunks = []

        for document in documents:
            chunks = self.chunker.chunk_document(document)
            all_chunks.extend(chunks)

        return all_chunks

    def index(self, recreate: bool = False) -> int:
        """
        Index all document chunks into Qdrant.

        Args:
            recreate: Delete and recreate the collection first.

        Returns:
            Number of indexed chunks.
        """

        self.vector_store.create_collection(
            recreate=recreate
        )

        chunks = self.build_chunks()

        if not chunks:
            return 0

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
        )

        points = []

        for idx, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "filename": chunk["filename"],
                        "file_type": chunk["file_type"],
                        "chunk_index": chunk["chunk_index"],
                    },
                )
            )

        self.vector_store.upsert(points)

        return len(points)
