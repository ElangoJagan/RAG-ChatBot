from pathlib import Path
import logging
from dataclasses import dataclass, field
import hashlib

@dataclass
class Document:
    doc_id: str
    source: str
    page_content: str
    metadata: dict = field(default_factory=dict)

def _hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FileLoader:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def load(self):
        for file in self.directory.iterdir():
            if file.suffix == '.txt':
                text = Path(file).read_text(encoding='utf-8')
                yield Document(
                    doc_id = _hash_id(file.name + text[:100]),
                    source = f'file : {file.name}',
                    page_content = text, 
                    metadata = {'file_type':'txt'}
                    
                )
