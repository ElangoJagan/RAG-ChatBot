from sentence_transformers import SentenceTransformer



class Embedder:
    def __init__(self, model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self,texts):
        return self.model.encode(texts,convert_to_numpy=True, normalize_embeddings= True)
    
import faiss
import pickle
from pathlib import Path

class FAISSVectorStore:
    def __init__(self,dim = 384):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        
    def add(self,chunks, embeddings):
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
    
    def search(self,query_embedding,top_k=5):
        scores, indices = self.index.search(query_embedding.reshape(1,-1).astype('float32'), top_k)
        results= []
        for score,idx in zip(scores[0], indices[0]):
            if idx ==-1:
                continue 
            results.append((self.chunks[idx], float(score)))
            return results
    def save(self,path):
        Path(path).mkdir(parents= True, exist_ok=True)
        faiss.write_index(self.index, f'{path}/index.faiss')
        with open(f'{path}/chunks.pkl', 'wb')as f:
            pickle.dump(self.chunks, f)
    
    def load(self,path):
        self.index = faiss.read_index(f'{path}/index.faiss')
        with open (f'{path}/ chunks.pkl', "rb") as f:
            self.chunks = pickle.load(f)
        