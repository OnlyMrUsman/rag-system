from backend.core.ingestion.document_loader import DocumentLoader
from backend.core.ingestion.chunker import TextChunker


loader = DocumentLoader("data/documents")

documents = loader.load_all()

print(f"Documents found: {len(documents)}")

for document in documents:
    print(f"\nSource: {document['source']}")
    print(f"Characters: {len(document['text'])}")

chunker = TextChunker(
    chunk_size=200,
    chunk_overlap=50,
)

all_chunks = []

for document in documents:
    chunks = chunker.chunk_document(document)
    all_chunks.extend(chunks)

print(f"\nTotal chunks: {len(all_chunks)}")

for chunk in all_chunks:
    print(
        f"\nChunk {chunk['chunk_index']} "
        f"({chunk['filename']}):"
    )
    print(chunk["text"])

print("\nDocument ingestion test: PASSED")
