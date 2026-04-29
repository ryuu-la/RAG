# RAG Studio

Local-first **Agentic Retrieval-Augmented Generation (RAG)** application: upload PDFs, chunk & embed them into **ChromaDB**, retrieve with dense + **BM25** hybrid search, and answer with **Google Generative AI** models. Features a modern **React + Vite** chat UI with multi-turn memory, agent reasoning, web search, document export, and real-time citations.

**Repository:** [github.com/ryuu-la/RAG](https://github.com/ryuu-la/RAG)

---

## App Screenshots

<p align="center">
  <img src="ASSETS/Screenshot 2026-04-29 220633.png" alt="RAG Studio — Dark Mode Landing" width="100%" />
  <br/>
  <em>Landing Page — Dark Theme</em>
</p>

<p align="center">
  <img src="ASSETS/Screenshot 2026-04-29 220538.png" alt="RAG Studio — Chat Response with Agent Reasoning" width="100%" />
  <br/>
  <em>Chat Response with Agent Reasoning Steps — Light Theme</em>
</p>

---

## Features

- **Agentic RAG** — AI agent with tool access: document search, web search, URL reading, PDF/CSV export
- **Multi-turn conversation memory** — model remembers previous messages within a session
- **Hybrid retrieval** — dense semantic search + BM25 keyword search for best results
- **PDF ingestion** with token/chunk metrics and OCR fallback (Tesseract)
- **Web search** via DuckDuckGo for real-time information
- **Smart search routing** — auto-routes to index or web based on query intent
- **Dual citation system** — web URLs for web searches, document references for index searches
- **PDF & CSV export** — generate professional documents and spreadsheets from AI responses
- **"Upload to RAG"** (indexed) vs **"Upload to Model"** (direct context) flows
- **Dark/Light theme** toggle
- **Session management** — multiple chat sessions with history

## Tech Stack

| Layer | Stack |
|--------|--------|
| Backend | Python 3.11+, FastAPI, ChromaDB, sentence-transformers, rank-bm25, LangChain |
| Frontend | React 18, Vite 5, react-markdown |
| LLM | Google Generative AI (`GOOGLE_API_KEY`) — Gemma 4 31B, Gemini 3.1 Flash Lite |
| Search | DuckDuckGo (web), Hybrid dense+BM25 (documents) |

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** (20+ recommended)
3. **Git**
4. **Google API Key** — get one from [Google AI Studio](https://aistudio.google.com/apikey)
5. **Tesseract OCR** *(optional, for scanned/image-heavy PDFs)*
   - Windows: install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and ensure `tesseract` is on `PATH`.

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

### 3. Environment Variables

Copy the example env and add your **Google API key**:

**Windows (from `RAG/`):**

```powershell
copy .env.example backend\.env
```

**macOS / Linux (from `RAG/`):**

```bash
cp .env.example backend/.env
```

Edit **`backend/.env`**. Minimum required:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

See **`.env.example`** at the repo root for chunking, OCR, paths, and embedding model options.

### 4. Frontend (Node)

From **`RAG/frontend`**:

```bash
npm install
```

---

## How to Run

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

### Production Build (Frontend Only)

```bash
cd frontend
npm run build
```

Static output is in `frontend/dist/`.

---

## Using the App

1. **Upload to RAG** — PDF is parsed, chunked, embedded, and stored in Chroma for retrieval.
2. **Upload to Model** — file is used as direct model context (not indexed in the vector DB).
3. **Ask questions** in the chat — the agent automatically decides whether to search the web or your indexed documents.
4. **Multi-turn conversations** — follow-up questions reference previous context within the same session.
5. **Export results** — ask the agent to generate PDF reports or CSV spreadsheets from its responses.
6. Open the right **Stats** panel for documents, chunks, and retrieval metrics.
7. First embedding run downloads **`EMBEDDING_MODEL`** into `backend/data/models/` (can take a few minutes on CPU).

---

## API Overview

Base URL: `http://localhost:8001`

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/api/models` | List available LLM models |
| POST | `/api/ingest/upload` | Upload PDF for RAG indexing |
| GET | `/api/ingest/status/{job_id}` | Ingest job status |
| GET | `/api/sources` | List indexed sources |
| DELETE | `/api/documents/{doc_id}` | Delete a document |
| POST | `/api/model/upload` | Direct model upload |
| GET | `/api/model/uploads` | List model uploads |
| POST | `/api/query` | RAG / chat query (non-streaming) |
| POST | `/api/query/stream` | Agentic RAG query (SSE streaming) |
| GET | `/api/exports/{filename}` | Download exported files |
| GET | `/api/metrics/...` | Metrics helpers |

---

## OCR

- Controlled by `OCR_ENABLED` and `OCR_MIN_CHARS_PER_PAGE` in `.env`.
- If Tesseract is missing, text-based PDFs still work; OCR path is skipped.

---

## Embedding Model (CPU-friendly)

For typical **CPU-only** setups, **`BAAI/bge-small-en-v1.5`** (default) is a good balance of quality and speed. Change `EMBEDDING_MODEL` in `.env` if needed; weights stay under `backend/data/models/` locally.

---

## Troubleshooting

| Issue | What to check |
|--------|----------------|
| `401` / LLM errors | `GOOGLE_API_KEY` in `backend/.env` |
| Frontend can't reach API | Backend running; `VITE_API_BASE` matches port (`8001`) |
| CORS | Backend is configured for local dev; use same machine/ports as in docs |
| Slow first query | First run downloads embedding model to `backend/data/models/` |
| OCR errors | Install Tesseract and add to `PATH` |

---

## License

This project is for educational and personal use.
