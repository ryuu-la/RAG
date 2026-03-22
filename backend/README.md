# Backend (FastAPI)

## Run Locally

1. Create virtual environment:
   - `python -m venv .venv`
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create `backend/.env` from repo root `.env.example` and set `GROQ_API_KEY`.
4. Start server (port should match frontend `VITE_API_BASE`, default **8001**):
   - `uvicorn app.main:app --reload --host 0.0.0.0 --port 8001`

## Current Endpoints

- `GET /health`
- `POST /api/ingest/upload`
- `GET /api/ingest/status/{job_id}`
- `GET /api/sources`
- `POST /api/model/upload`
- `GET /api/model/uploads`
- `GET /api/metrics/document/{doc_id}`
- `POST /api/query`
- `GET /api/metrics/query/{query_id}`

## Notes

- PDF upload + parsing + token/page/chunk metrics are available.
- Chunks are indexed into local ChromaDB and queried with hybrid retrieval (dense + BM25 fusion).
- Groq API answer generation is wired with citations.
- OCR fallback is enabled for low-text PDF pages when Tesseract is installed.
- Default LLM is configured via `GROQ_MODEL` in `.env` (see `.env.example` in repo root).
- Recommended embedding model for this machine class: `BAAI/bge-small-en-v1.5`
- Model cache/download path is forced to `RAG/data/models` via:
  - `HF_HOME`
  - `TRANSFORMERS_CACHE`
  - `SENTENCE_TRANSFORMERS_HOME`

## Where files are stored

- Uploaded PDFs: `RAG/data/uploads`
- Chroma vector data: `RAG/data/chroma`
- Embedding model cache: `RAG/data/models`
- Direct model uploads: `RAG/data/model_uploads`
