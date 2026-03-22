from __future__ import annotations

import uuid
from pathlib import Path

from pypdf import PdfReader

from app.config import settings
from app.services.ingest import estimate_tokens
from app.store import store


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError("Unsupported file for direct model upload. Use PDF, TXT, or MD.")


def save_model_upload(filename: str, content: bytes) -> dict[str, str | int]:
    upload_id = str(uuid.uuid4())
    model_upload_dir = settings.data_path / "model_uploads"
    model_upload_dir.mkdir(parents=True, exist_ok=True)
    path = model_upload_dir / f"{upload_id}_{filename}"
    path.write_bytes(content)

    text = _extract_text(path)
    tokens = estimate_tokens(text)
    store.model_uploads[upload_id] = {
        "upload_id": upload_id,
        "filename": filename,
        "path": str(path),
        "text": text,
        "estimated_tokens": tokens,
    }
    return {"upload_id": upload_id, "estimated_tokens": tokens}


def build_model_upload_context(upload_ids: list[str] | None) -> str:
    if not upload_ids:
        return ""
    blocks: list[str] = []
    for uid in upload_ids:
        item = store.model_uploads.get(uid)
        if not item:
            continue
        blocks.append(f"[MODEL_FILE] filename={item['filename']}\n{item['text']}")
    return "\n\n".join(blocks)
