from rank_bm25 import BM250kapi
import numpy as np 


class HybridRetriever:
    def __init__(self, vector_store, embedder, chunks, vector_weight=0.6, bm25_weight=0.4):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunks = chunks
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        tokenized_corpus = [c.text.lower().split() for c in chunks]
        self.bm25 = BM250kapi(tokenized_corpus)
    
    def retrieve(self, query, top_k = 5, candidate_pool=15):
        #1.vector search
        query_emb = self.embedder.embed([query])[0]
        vector_results = self.vector_store.search(query_emb, top_k = candidate_pool)
        vector_scores = {chunk.chunk_id: score for chunk, score in vector_results}
        
        #2.BM25 search
        bm25_scores_all = self.bm25.get_scores(query.lower().split())
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:candidate_pool]
        bm25_scores = {self.chunks[i].chunk_id: bm25_scores_all[i] for i in  top_bm25_idx}
        
        #3 fuse both score sets
        all_ids = set(vector_scores) | set(bm25_scores)
        fused = self._fuse(all_ids, vector_scores, bm25_scores)
        
        #4 sort by  fused score , take top k 
        top_ids =[cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])[:top_k]]
        chunk_by_id= {c.chunk_id:c for  c in  self.chunks}
        return [chunk_by_id[cid] for  cid in  top_ids]
    
    
    def _fuse(self, all_ids, vec, bm25):
        def normalize(d):
            if not d:
                return {}
            vals = list(d.values())
            lo,hi = min(vals), max(vals)
            if hi ==lo:
                return {k:1.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}
        
        vec_n, bm25_n = normalize(vec), normalize(bm25)
        return{
            cid:self.vector_weight*vec_n.get(cid,0)+self.bm25_weight*bm25_n.get(cid,0)
            for cid in all_ids
        }