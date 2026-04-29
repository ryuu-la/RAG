from __future__ import annotations

import time
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import settings
from app.models import (
    Citation,
    DocumentMetricsResponse,
    IngestStatusResponse,
    ModelUploadInfo,
    QueryMetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from app.services.ingest import create_ingest_job, ensure_dirs, process_pdf
from app.services.llm import generate_answer
from app.services.model_upload import build_model_upload_context, save_model_upload
from app.services.retrieval import (
    delete_document_chunks,
    hybrid_search,
    rebuild_bm25_index,
    restore_documents_from_chroma,
)
from app.services.agent import run_agent_stream, AVAILABLE_MODELS
from app.store import store

app = FastAPI(title="RAG Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_dirs()
    settings.export_path.mkdir(parents=True, exist_ok=True)
    restored = restore_documents_from_chroma()
    for doc_id, doc_meta in restored.items():
        if doc_id not in store.documents:
            store.documents[doc_id] = doc_meta
    rebuild_bm25_index()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def list_models() -> list[dict]:
    return AVAILABLE_MODELS


# ── Ingest ──


@app.post("/api/ingest/upload")
async def upload_file(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF upload is supported currently.")

    doc_id = str(uuid.uuid4())
    job_id = create_ingest_job(file.filename)

    saved_path = settings.upload_path / f"{doc_id}_{file.filename}"
    content = await file.read()
    saved_path.write_bytes(content)

    thread = threading.Thread(
        target=process_pdf,
        kwargs={"job_id": job_id, "saved_path": saved_path, "doc_id": doc_id},
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "doc_id": doc_id}


@app.get("/api/ingest/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str) -> IngestStatusResponse:
    if job_id not in store.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    data = store.jobs[job_id]
    return IngestStatusResponse(
        job_id=job_id,
        state=data["state"],
        message=data.get("message", ""),
        progress=data.get("progress", 0),
    )


# ── Sources / Documents ──


@app.get("/api/sources", response_model=list[SourceInfo])
def list_sources() -> list[SourceInfo]:
    return [
        SourceInfo(doc_id=d["doc_id"], filename=d["filename"], page_count=d.get("page_count"))
        for d in store.documents.values()
    ]


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str | int]:
    doc = store.documents.pop(doc_id, None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    deleted_chunks = delete_document_chunks(doc_id)
    path = doc.get("path")
    if path:
        try:
            import os
            os.remove(path)
        except OSError:
            pass
    return {"doc_id": doc_id, "deleted_chunks": deleted_chunks}


# ── Model uploads ──


@app.post("/api/model/upload")
async def model_upload(file: UploadFile = File(...)) -> dict[str, str | int]:
    allowed = (".pdf", ".txt", ".md")
    if not file.filename or not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail="Model upload supports PDF, TXT, and MD.")
    content = await file.read()
    return save_model_upload(file.filename, content)


@app.get("/api/model/uploads", response_model=list[ModelUploadInfo])
def list_model_uploads() -> list[ModelUploadInfo]:
    return [
        ModelUploadInfo(
            upload_id=item["upload_id"],
            filename=item["filename"],
            estimated_tokens=item["estimated_tokens"],
        )
        for item in store.model_uploads.values()
    ]


# ── Metrics ──


@app.get("/api/metrics/document/{doc_id}", response_model=DocumentMetricsResponse)
def document_metrics(doc_id: str) -> DocumentMetricsResponse:
    doc = store.documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentMetricsResponse(**doc)


# ── Query (legacy, non-streaming) ──


@app.post("/api/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    top_k = payload.top_k or settings.top_k

    chunks = hybrid_search(payload.question, top_k=top_k)
    model_file_context = build_model_upload_context(payload.model_upload_ids)
    answer, response_tokens, context_tokens = generate_answer(
        payload.question,
        chunks,
        extra_context=model_file_context,
        model_override=payload.selected_model,
    )

    query_id = str(uuid.uuid4())
    store.query_metrics[query_id] = {
        "query_id": query_id,
        "retrieved_chunks": len(chunks),
        "context_tokens_sent": context_tokens,
        "response_tokens": response_tokens,
    }
    citations = [
        Citation(
            source=(chunk.get("metadata", {}) or {}).get("source", "unknown.pdf"),
            page=(chunk.get("metadata", {}) or {}).get("page"),
            chunk_id=(chunk.get("metadata", {}) or {}).get("chunk_id", chunk.get("chunk_id", "na")),
        )
        for chunk in chunks
    ]
    latency_ms = int((time.perf_counter() - start) * 1000)
    return QueryResponse(query_id=query_id, answer=answer, citations=citations, latency_ms=latency_ms)


# ── Query (SSE streaming agent) ──


@app.post("/api/query/stream")
async def query_stream(payload: QueryRequest):
    model_file_context = build_model_upload_context(payload.model_upload_ids)

    async def event_gen():
        history_dicts = None
        if payload.history:
            history_dicts = [{"role": m.role, "content": m.content} for m in payload.history]

        async for event in run_agent_stream(
            question=payload.question,
            rag_mode=payload.rag_mode,
            model_name=payload.selected_model,
            extra_context=model_file_context,
            history=history_dicts,
        ):
            yield event

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Exports download ──


@app.get("/api/exports/{filename}")
def download_export(filename: str):
    safe_name = Path(filename).name
    path = settings.export_path / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    return FileResponse(str(path), filename=safe_name)


# ── Query metrics ──


@app.get("/api/metrics/query/{query_id}", response_model=QueryMetricsResponse)
def query_metrics(query_id: str) -> QueryMetricsResponse:
    data = store.query_metrics.get(query_id)
    if not data:
        raise HTTPException(status_code=404, detail="Query metrics not found")
    return QueryMetricsResponse(**data)
