from typing import List, Dict


class TextChunker:
    """Split documents into overlapping text chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text using character-based overlapping windows."""
        text = " ".join(text.split())

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            start = end - self.chunk_overlap

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        """Chunk one loaded document while preserving metadata."""
        chunks = self.split_text(document["text"])

        return [
            {
                "text": chunk,
                "source": document["source"],
                "filename": document["filename"],
                "file_type": document["file_type"],
                "chunk_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]
