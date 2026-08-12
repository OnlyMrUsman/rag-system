from sentence_transformers import SentenceTransformer
import torch

print("CUDA:", torch.cuda.is_available())

model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cuda"
)

embedding = model.encode(
    ["Hello world"]
)

print("Embedding shape:", embedding.shape)
