from src.retrieval.hybrid_retriver import HybridRetriever

retriever = HybridRetriever(store, embedder, all_chunks)
results = retriever.retrieve("How many vacation days do employees get?", top_k=3)
for c in results:
    print(c.source, "-", c.text[:100])