from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import settings

_embedder: SentenceTransformer | None = None
_collection: Any | None = None
_bm25: BM25Okapi | None = None
_bm25_chunk_ids: list[str] = []
_bm25_docs: dict[str, dict[str, Any]] = {}


def _tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def configure_local_model_cache() -> None:
    # Keep all downloaded HF/SentenceTransformer assets inside project folder.
    model_cache = str(settings.model_cache_path.resolve())
    os.environ["HF_HOME"] = model_cache
    os.environ["TRANSFORMERS_CACHE"] = model_cache
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_cache


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        configure_local_model_cache()
        settings.model_cache_path.mkdir(parents=True, exist_ok=True)
        _embedder = SentenceTransformer(
            settings.embedding_model,
            cache_folder=str(settings.model_cache_path.resolve()),
        )
    return _embedder


def get_collection() -> Any:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(settings.chroma_path.resolve()))
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def index_chunks(chunks: list[dict[str, Any]], rebuild_bm25: bool = False) -> int:
    if not chunks:
        return 0
    collection = get_collection()
    ids = [chunk["chunk_id"] for chunk in chunks]
    docs = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "doc_id": chunk["doc_id"],
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_id": chunk["chunk_id"],
        }
        for chunk in chunks
    ]
    embeddings = embed_texts(docs)
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    if rebuild_bm25:
        rebuild_bm25_index()
    return len(ids)


def delete_document_chunks(doc_id: str) -> int:
    collection = get_collection()
    raw = collection.get(where={"doc_id": doc_id}, include=[])
    ids = raw.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    rebuild_bm25_index()
    return len(ids)


def restore_documents_from_chroma() -> dict[str, dict[str, Any]]:
    """Reconstruct document metadata from persisted ChromaDB chunks on startup."""
    collection = get_collection()
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])
    docs_map: dict[str, dict[str, Any]] = {}
    for chunk_id, text, meta in zip(ids, docs, metas):
        if not meta:
            continue
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("source", "unknown.pdf"),
                "path": "",
                "page_count": 0,
                "pages_with_text": 0,
                "pages_ocr_used": 0,
                "estimated_tokens": 0,
                "chunk_count": 0,
                "indexed_chunk_count": 0,
                "_pages": set(),
            }
        entry = docs_map[doc_id]
        entry["chunk_count"] += 1
        entry["indexed_chunk_count"] += 1
        entry["estimated_tokens"] += max(1, int(len(text or "") / 4))
        page = meta.get("page", 0)
        if page:
            entry["_pages"].add(page)
    for entry in docs_map.values():
        entry["page_count"] = len(entry["_pages"]) if entry["_pages"] else 0
        del entry["_pages"]
    return docs_map


def rebuild_bm25_index() -> None:
    global _bm25, _bm25_chunk_ids, _bm25_docs
    collection = get_collection()
    raw = collection.get(include=["documents", "metadatas"])
    ids = raw.get("ids", [])
    docs = raw.get("documents", [])
    metas = raw.get("metadatas", [])
    tokenized: list[list[str]] = []
    _bm25_docs = {}
    _bm25_chunk_ids = []
    for chunk_id, text, meta in zip(ids, docs, metas):
        tokens = _tokenize_for_bm25(text or "")
        tokenized.append(tokens if tokens else ["_"])
        _bm25_chunk_ids.append(chunk_id)
        _bm25_docs[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text or "",
            "metadata": meta or {},
        }
    _bm25 = BM25Okapi(tokenized) if tokenized else None


def hybrid_search(query: str, top_k: int) -> list[dict[str, Any]]:
    collection = get_collection()
    query_embedding = embed_texts([query])[0]
    dense_n = max(top_k * 4, 8)
    dense_raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=dense_n,
        include=["documents", "metadatas", "distances"],
    )

    dense_docs = dense_raw.get("documents", [[]])[0]
    dense_metas = dense_raw.get("metadatas", [[]])[0]
    dense_dists = dense_raw.get("distances", [[]])[0]
    dense_ids = [meta.get("chunk_id") for meta in dense_metas]

    # Reciprocal rank fusion between dense and BM25 ranks.
    fused = defaultdict(float)
    dense_data: dict[str, dict[str, Any]] = {}
    for rank, (chunk_id, text, meta, dist) in enumerate(zip(dense_ids, dense_docs, dense_metas, dense_dists), start=1):
        if not chunk_id:
            continue
        fused[chunk_id] += 1.0 / (60 + rank)
        dense_data[chunk_id] = {"chunk_id": chunk_id, "text": text, "metadata": meta, "distance": dist}

    if _bm25 is not None and _bm25_chunk_ids:
        q_tokens = _tokenize_for_bm25(query)
        scores = _bm25.get_scores(q_tokens if q_tokens else ["_"])
        if len(scores) > 0:
            candidate_size = min(len(scores), max(top_k * 6, 20))
            top_idx = np.argpartition(scores, -candidate_size)[-candidate_size:]
            sorted_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            for rank, idx in enumerate(sorted_idx.tolist(), start=1):
                chunk_id = _bm25_chunk_ids[idx]
                fused[chunk_id] += 1.0 / (60 + rank)

    ranked_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)[:top_k]
    results: list[dict[str, Any]] = []
    for cid in ranked_ids:
        if cid in dense_data:
            results.append(dense_data[cid])
        elif cid in _bm25_docs:
            bm = _bm25_docs[cid]
            results.append({"chunk_id": cid, "text": bm["text"], "metadata": bm["metadata"], "distance": None})
    return results
