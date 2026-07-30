# HR RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions over
HR policy documents — built from scratch to reflect real production
patterns: multi-format ingestion, semantic chunking, hybrid retrieval,
and an LLM-backed API, with CI automated on every push.

## What it does

Ask a question like *"How many vacation days do employees get?"* and the
system retrieves the most relevant passages from ingested HR documents
(PDF, DOCX, TXT), then uses an LLM (Groq/Llama 3.3) to generate a grounded
answer with source attribution — instead of relying on the LLM's own
(unverified) knowledge.

## Architecture

```
data/raw/ (.txt, .pdf, .docx)
        │
        ▼
  FileLoader ──────────► Document objects (common shape, per-file error handling)
        │
        ▼
 SemanticChunker ───────► Chunk objects (paragraph-aware splitting + overlap)
        │
        ▼
   Embedder ─────────────► sentence-transformers (all-MiniLM-L6-v2, 384-dim)
        │
        ▼
 FAISSVectorStore ──────► in-memory vector index (cosine similarity via inner product)
        │
        ▼
 HybridRetriever ───────► BM25 (keyword) + vector search, fused by weighted score
        │
        ▼
   FastAPI /query ──────► retrieves top chunks → Groq LLM → grounded answer + sources
```

## Why these design choices

- **Hybrid retrieval, not just vector search** — pure vector similarity
  misses exact keyword/ID matches (e.g. a specific policy number). BM25
  catches those; fusing both with normalized, weighted scores gives better
  coverage than either alone.
- **Paragraph-aware chunking with overlap** — splitting text blindly by
  character count cuts sentences in half and destroys meaning. Chunks are
  built from paragraph boundaries, with a small character overlap between
  consecutive chunks so a fact sitting near a cut point isn't lost from
  context.
- **Per-file error handling in ingestion** — one corrupted or unreadable
  file (tested deliberately with a broken PDF) logs an error and is
  skipped, without crashing the rest of the ingestion batch. This matters
  at real document volume, where some fraction of files are always bad.
- **Common `Document`/`Chunk` data shapes** — every ingestion source
  (currently file-based; web/API ingestion designed but not yet built)
  normalizes into the same object shape, so chunking/embedding/retrieval
  code never needs to know where the data originated.

## Project structure

```
src/
  ingestion/
    file_loader.py       # Document model + FileLoader (.txt, .pdf, .docx)
  processing/
    chunking.py           # Chunk model + SemanticChunker
  embeddings/
    embedder.py            # Embedder (sentence-transformers) + FAISSVectorStore
  retrieval/
    hybrid_retriever.py     # BM25 + vector fusion retrieval
  api/
    main.py                  # FastAPI service: builds index at startup, /query, /health
tests/
  test_chunking.py           # unit tests, run automatically in CI
docker/
  Dockerfile                  # written, not yet locally verified (see Known Limitations)
docker-compose.yaml
.github/workflows/
  ci.yml                       # lint-free test automation, runs on every push
pytest.ini
requirements.txt
data/raw/                       # source HR documents (.txt / .pdf / .docx)
```

## Running locally

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# set your Groq API key (free tier: console.groq.com)
$env:GROQ_API_KEY="your_key_here"

uvicorn src.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, try the `/query` endpoint with a real question.

## Running tests

```powershell
pytest tests/ -v
```

CI runs this automatically on every push via GitHub Actions
(`.github/workflows/ci.yml`).

## Known limitations — stated upfront, not hidden

- **CI only, not yet CD.** The GitHub Actions workflow runs tests on push;
  it does not yet build/push a Docker image or deploy anywhere. A build
  step is the natural next addition once Docker is verified locally.
- **Docker written but not locally run.** `docker/Dockerfile` and
  `docker-compose.yaml` are complete and reviewable, but haven't been
  build-tested on this machine due to a local virtualization/WSL2 issue.
  Pending environment fix.
- **Single-source ingestion.** Only file-based ingestion (`.txt`, `.pdf`,
  `.docx`) is implemented. Web scraping and REST API ingestion were
  designed into the same `Document` shape but not yet built — the
  architecture supports adding them without touching downstream code.
- **No automated evaluation harness yet.** Retrieval/answer quality has
  been spot-checked manually (real queries against real HR document
  content), not measured with a formal metric (e.g. RAGAS-style
  precision/recall/faithfulness scoring). This is the next planned
  addition, and is the piece most portfolio RAG projects skip entirely.
- **In-memory FAISS index, rebuilt on every API startup.** Fine at this
  scale; wouldn't hold up to a large corpus or fast restarts in
  production — a persisted/loadable index (`save`/`load`, already
  supported by `FAISSVectorStore`) would be the fix.
- **No auth, rate-limiting, or monitoring** on the API yet.

## What was actually debugged building this (worth knowing for interviews)

- A mutable-default-argument bug in the `Document`/`Chunk` dataclasses
  (`metadata: dict = {}` sharing state across instances) — fixed with
  `field(default_factory=dict)`.
- A silent content-loss bug in chunking, where short-but-legitimate text
  was dropped instead of merged when below `min_chunk_size`.
- Windows-specific encoding issues (BOM characters from PowerShell/Notepad
  file creation) breaking both a GitHub Actions YAML file and `pytest.ini`
  — fixed by recreating files with an editor that saves clean UTF-8.
- A broken-PDF test proving per-file error handling actually works, rather
  than assuming it does.