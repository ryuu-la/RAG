# RAG Studio

Local-first **Retrieval-Augmented Generation (RAG)** app: upload PDFs, chunk & embed them into **ChromaDB**, retrieve with dense + **BM25** hybrid search, and answer with **Groq** LLM APIs. Includes a **React + Vite** UI (chat, uploads, metrics, citations).

**Repository:** [github.com/ryuu-la/RAG](https://github.com/ryuu-la/RAG)

## Features

- PDF ingestion (large documents supported) with token/chunk metrics
- Text extraction with **OCR fallback** (Tesseract) on low-text pages
- Sentence-transformers embeddings (default: `BAAI/bge-small-en-v1.5`) — cached under `backend/data/models` (not in Git)
- Persistent Chroma vector store under `backend/data/chroma`
- FastAPI backend + modern chat UI
- “Upload to RAG” (indexed) vs “Upload to Model” (direct context) flows

## Tech stack

| Layer | Stack |
|--------|--------|
| Backend | Python 3.11+, FastAPI, ChromaDB, sentence-transformers, rank-bm25, Groq |
| Frontend | React 18, Vite 5 |
| LLM | Groq API (`GROQ_API_KEY`, `GROQ_MODEL`) |

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** (20+ recommended)
3. **Git**
4. **Tesseract OCR** (optional but recommended for scanned/image-heavy PDFs)  
   - Windows: install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract` is on `PATH`.

## What gets committed vs stays local

| Committed to GitHub | **Not** committed (see `.gitignore`) |
|---------------------|----------------------------------------|
| Source code, `requirements.txt`, `package.json` / lockfile | `.env` / `backend/.env` (**API keys**) |
| `.env.example` (no secrets) | `backend/data/models/` (**embedding weights**) |
| Docs | `backend/data/uploads/`, `backend/data/chroma/`, `backend/data/model_uploads/` (local data) |
| | `node_modules/`, `frontend/dist/`, virtualenvs |

Clone the repo, then copy env and let the app download models on first run.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/ryuu-la/RAG.git
cd RAG
```

### 2. Backend (Python)

From **`RAG/backend`**:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment variables

Copy the example env and add your **Groq API key**:

**Windows (from `RAG/`):**

```powershell
copy .env.example backend\.env
```

**macOS / Linux (from `RAG/`):**

```bash
cp .env.example backend/.env
```

Edit **`backend/.env`** (this file is gitignored). Minimum:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

See **`.env.example`** at the repo root for chunking, OCR, paths, and embedding model options.

### 4. Frontend (Node)

From **`RAG/frontend`**:

```bash
npm install
```

---

## How to run

Use **two terminals**: one for the API, one for the UI.

### Terminal A — Backend

From **`RAG/backend`** (with venv activated):

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Default API URL: **`http://localhost:8001`**

### Terminal B — Frontend

From **`RAG/frontend`**:

**Windows (PowerShell)** — point the UI at the API:

```powershell
$env:VITE_API_BASE="http://localhost:8001"
npm run dev -- --host 0.0.0.0 --port 5173
```

**macOS / Linux**

```bash
VITE_API_BASE=http://localhost:8001 npm run dev -- --host 0.0.0.0 --port 5173
```

Open **`http://localhost:5173`** in the browser.

> If you omit `VITE_API_BASE`, the frontend defaults to `http://localhost:8001` (see `frontend/src/api.js`).

### Production build (frontend only)

```bash
cd frontend
npm run build
```

Static output is in `frontend/dist/` (ignored by Git).

---

## Using the app

1. **Upload to RAG** — PDF is parsed, chunked, embedded, and stored in Chroma.
2. **Upload to Model** — file is used as direct model context (not indexed in the vector DB).
3. Ask questions in the chat; open the right **Stats** panel for documents, chunks, and last-query retrieval metrics.
4. First embedding run downloads **`EMBEDDING_MODEL`** into `backend/data/models/` (can take a few minutes on CPU).

---

## API overview

Base URL: `http://localhost:8001` (if using port above)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/api/ingest/upload` | Upload PDF for RAG indexing |
| GET | `/api/ingest/status/{job_id}` | Ingest job status |
| GET | `/api/sources` | List indexed sources |
| POST | `/api/model/upload` | Direct model upload |
| GET | `/api/model/uploads` | List model uploads |
| POST | `/api/query` | RAG / chat query |
| GET | `/api/metrics/...` | Metrics helpers |

More detail: `backend/README.md`.

---

## OCR

- Controlled by `OCR_ENABLED` and `OCR_MIN_CHARS_PER_PAGE` in `.env`.
- If Tesseract is missing, text-based PDFs still work; OCR path is skipped.

---

## Embedding model (CPU-friendly demo)

For typical **CPU-only** setups, **`BAAI/bge-small-en-v1.5`** (default) is a good balance of quality and speed. Change `EMBEDDING_MODEL` in `.env` if needed; weights stay under `backend/data/models/` locally.

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `401` / LLM errors | `GROQ_API_KEY` in `backend/.env` |
| Frontend can’t reach API | Backend running; `VITE_API_BASE` matches port (`8001`) |
| CORS | Backend is configured for local dev; use same machine/ports as in docs |
| Slow first query | First run downloads embedding model to `backend/data/models/` |
| OCR errors | Install Tesseract and `PATH` |

---

## License

Add a license file if you plan to open-source formally (e.g. MIT).
