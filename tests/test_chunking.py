from src.ingestion.file_loader import Document
from src.processing.chunking import SemanticChunker




def test_short_document_single_chunk():
    chunker = SemanticChunker(max_chunk_size=800)
    doc = Document(doc_id="t1", source="file:test.txt", page_content="Employees are entitled to eighteen days of paid annual leave per calendar year, to be used at their discretion with manager approval.")
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1


def test_long_document_splits_into_multiple_chunks():
    chunker = SemanticChunker(max_chunk_size=100, overlap=20)
    paragraphs = "\n\n".join([f"Paragraph number {i} with some content here." for i in range(10)])
    doc = Document(doc_id="t2", source="file:test.txt", page_content=paragraphs)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1