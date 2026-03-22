# Frontend (React + Vite)

## Features

- Sidebar chat history
- Modern chat area with citations
- Bottom composer with:
  - model selector
  - plus button with:
    - Upload to RAG
    - Upload to Model
- Metrics panel:
  - total documents
  - total tokens
  - chunks and indexed chunks
  - direct model uploads and token counts
  - last query context/response token stats

## Run

1. Install dependencies:
   - `npm install`
2. Start dev server:
   - `npm run dev`

Optional API base:

- Set `VITE_API_BASE` (default: `http://localhost:8000`)
