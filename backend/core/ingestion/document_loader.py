from pathlib import Path
from typing import List, Dict


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
}


class DocumentLoader:
    """Load supported documents from the RAG document directory."""

    def __init__(self, documents_dir: str = "data/documents"):
        self.documents_dir = Path(documents_dir)

    def discover_files(self) -> List[Path]:
        """Find all supported documents."""
        if not self.documents_dir.exists():
            return []

        return sorted(
            path
            for path in self.documents_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load_text_file(self, path: Path) -> str:
        """Load TXT or Markdown files."""
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    def load(self, path: Path) -> Dict:
        """Load a document and return normalized metadata."""
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md"}:
            text = self.load_text_file(path)

        else:
            raise NotImplementedError(
                f"Loader for {suffix} is not implemented yet."
            )

        return {
            "text": text,
            "source": str(path),
            "filename": path.name,
            "file_type": suffix,
        }

    def load_all(self) -> List[Dict]:
        """Load every supported document."""
        documents = []

        for path in self.discover_files():
            documents.append(self.load(path))

        return documents
