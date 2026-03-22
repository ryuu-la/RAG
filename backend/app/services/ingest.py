from __future__ import annotations

import re
import uuid
from pathlib import Path

from pypdf import PdfReader
from PIL import Image

try:
    import pypdfium2 as pdfium
except Exception:  # noqa: BLE001
    pdfium = None

try:
    import pytesseract
except Exception:  # noqa: BLE001
    pytesseract = None

from app.config import settings
from app.services.retrieval import index_chunks
from app.store import store


def ensure_dirs() -> None:
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.model_cache_path.mkdir(parents=True, exist_ok=True)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def create_ingest_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    store.jobs[job_id] = {
        "state": "queued",
        "message": "Queued for ingestion",
        "filename": filename,
        "progress": 0,
    }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    if job_id in store.jobs:
        store.jobs[job_id].update(kwargs)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_page_text(text: str, page: int, doc_id: str, source: str) -> list[dict]:
    chunks: list[dict] = []
    normalized = _normalize_text(text)
    if not normalized:
        return chunks

    size = max(settings.chunk_size, 200)
    overlap = min(max(settings.chunk_overlap, 0), size // 2)
    start = 0
    chunk_index = 0
    while start < len(normalized):
        end = min(start + size, len(normalized))
        chunk_text = normalized[start:end]
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )
            chunk_index += 1
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def _ocr_pdf_page(pdf_path: Path, page_index_zero_based: int) -> str:
    if not settings.ocr_enabled:
        return ""
    if pdfium is None or pytesseract is None:
        return ""
    try:
        doc = pdfium.PdfDocument(str(pdf_path))
        page = doc[page_index_zero_based]
        bitmap = page.render(scale=2.0)
        pil_image: Image.Image = bitmap.to_pil()
        text = pytesseract.image_to_string(pil_image) or ""
        return text
    except Exception:  # noqa: BLE001
        return ""


def process_pdf(job_id: str, saved_path: Path, doc_id: str) -> None:
    try:
        _update_job(job_id, state="processing", message="Reading PDF pages", progress=5)

        reader = PdfReader(str(saved_path))
        total_pages = len(reader.pages)
        page_texts: list[str] = []
        all_chunks: list[dict] = []
        pages_with_text = 0
        pages_ocr_used = 0

        for idx, page in enumerate(reader.pages, start=1):
            pct = 5 + int((idx / total_pages) * 50)
            _update_job(job_id, message=f"Extracting page {idx}/{total_pages}", progress=pct)

            text = page.extract_text() or ""
            normalized = _normalize_text(text)
            if len(normalized) < settings.ocr_min_chars_per_page:
                ocr_text = _ocr_pdf_page(saved_path, idx - 1)
                if _normalize_text(ocr_text):
                    text = f"{text}\n{ocr_text}".strip()
                    pages_ocr_used += 1

            page_texts.append(text)
            if _normalize_text(text):
                pages_with_text += 1
            all_chunks.extend(_chunk_page_text(text=text, page=idx, doc_id=doc_id, source=saved_path.name))

        full_text = "\n".join(page_texts).strip()
        page_count = len(page_texts)
        total_tokens = estimate_tokens(full_text)
        chunk_count = len(all_chunks)

        _update_job(job_id, message=f"Embedding & indexing {chunk_count} chunks", progress=60)

        batch_size = 256
        indexed_count = 0
        total_batches = max(1, (len(all_chunks) + batch_size - 1) // batch_size)
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            is_last = (i + batch_size) >= len(all_chunks)
            indexed_count += index_chunks(batch, rebuild_bm25=is_last)
            pct = 60 + int(((i + len(batch)) / max(len(all_chunks), 1)) * 35)
            _update_job(job_id, message=f"Indexed {indexed_count}/{chunk_count} chunks", progress=min(pct, 95))

        store.documents[doc_id] = {
            "doc_id": doc_id,
            "filename": saved_path.name,
            "path": str(saved_path),
            "page_count": page_count,
            "pages_with_text": pages_with_text,
            "pages_ocr_used": pages_ocr_used,
            "estimated_tokens": total_tokens,
            "chunk_count": chunk_count,
            "indexed_chunk_count": indexed_count,
        }

        _update_job(job_id, state="indexed", message=f"Done! {indexed_count} chunks indexed.", progress=100)
    except Exception as exc:  # noqa: BLE001
        _update_job(job_id, state="failed", message=f"Ingestion failed: {exc}", progress=0)
