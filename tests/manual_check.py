from src.ingestion.file_loader import FileLoader

loader = FileLoader("data/raw")

from pathlib import Path


for doc in loader.load():
    print(doc.source, "-", len(doc.page_content), "characters")