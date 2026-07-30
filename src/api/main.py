from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

from src.ingestion.file_loader import FileLoader
from src.processing.chunking import SemanticChunker
from src.embeddings.embedder import Embedder, FAISSVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever

app = FastAPI(title = 'HR RAG CHATBOT')

# Build the index once at startup (loaded into memory)
loader = FileLoader('data/raw')
chunker = SemanticChunker(max_chunk_size = 500, overlap = 50)
all_chunks = []
for doc in loader.load():
    all_chunks.extend(chunker.chunk_document(doc))

embedder = Embedder()
embeddings = embedder.embed([c.text for c in all_chunks])

store = FAISSVectorStore(dim = 384)
store.add(all_chunks, embeddings)

retriever = HybridRetriever(store, embedder, all_chunks)
llm_client = Groq()

class QueryRequest(BaseModel):
    question:str

@app.post('/query')
def query(req:QueryRequest):
    results = retriever.retrieve(req.question, top_k=5)
    context = "\n---\n".join(c.text for c in results)
    
    
    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer using only the context above."
        }]
    )
    
    return {
        'answer': response.choices[0].message.content, 
        'sources': list({c.source for c in results})
    }