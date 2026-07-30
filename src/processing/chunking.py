
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id:str
    doc_id: str
    text:str
    chunk_index:int
    source:str
    metadata:dict = field(default_factory = dict)
    

class SemanticChunker:
    def __init__(self,max_chunk_size = 800, overlap= 100, min_chunk_size = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        
    def _split_paragraphs(self, text):
        import re
        paras = re.split(r'\n\s*\n', text)
        result=[]
        for p in paras:
            p = p.strip()
            if not p:
                continue
            result.append(p)
        return result 
    
    def _pack_paragraphs(self,paragraphs):
        chunks=[]
        current=""
        
        for para in paragraphs:
            if len(current)+len(para)+1<= self.max_chunk_size:
                current =f'{current}\n{para}' if current else para
            else:
                if current:
                    chunks.append(current)
                overlap_text = current[-self.overlap:] if current else ""
                current = f'{overlap_text}\n{para}'.strip() if overlap_text else para
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def chunk_document(self,doc):
        paragraphs = self._split_paragraphs(doc.page_content)
        raw_chunks = self._pack_paragraphs(paragraphs)
        
        chunks= []
        for i, text in enumerate(raw_chunks):
            if len(text.strip())<self.min_chunk_size:
                continue
            chunks.append(Chunk(
                chunk_id = f"{doc.doc_id}_{i}",
                doc_id = doc.doc_id,
                text = text.strip(),
                chunk_index=i,
                source = doc.source,
                metadata = doc.metadata
                
            ))
        
        return chunks
    
        