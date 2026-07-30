from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker
from src.embeddings.embedder import Embedder, FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

# 1. Load
loader = FileLoader("data/raw")

# 2. Chunk
chunker = SemanticChunker(max_chunk_size=500, overlap=50)
all_chunks = []
for doc in loader.load():
    all_chunks.extend(chunker.chunk_document(doc))
print(f"Total chunks: {len(all_chunks)}")

# 3. Embed
embedder = Embedder()
texts = [c.text for c in all_chunks]
embeddings = embedder.embed(texts)
print(f"Embeddings shape: {embeddings.shape}")

# 4. Store
store = FAISSVectorStore(dim=384)
store.add(all_chunks, embeddings)

# 5. Hybrid retrieve
retriever = HybridRetriever(store, embedder, all_chunks)
results = retriever.retrieve("How many vacation days do employees get?", top_k=3)

for c in results:
    print(c.source, "-", c.text[:100])