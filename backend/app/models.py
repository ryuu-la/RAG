from typing import Literal

from pydantic import BaseModel, Field


JobState = Literal["queued", "processing", "indexed", "failed"]


class IngestStatusResponse(BaseModel):
    job_id: str
    state: JobState
    message: str = ""
    progress: int = 0


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = None
    selected_model: str | None = None
    model_upload_ids: list[str] | None = None
    rag_mode: bool = True
    history: list[ChatMessage] | None = None


class Citation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    latency_ms: int


class SourceInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int | None = None


class ModelUploadInfo(BaseModel):
    upload_id: str
    filename: str
    estimated_tokens: int


class DocumentMetricsResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    pages_with_text: int | None = None
    pages_ocr_used: int | None = None
    estimated_tokens: int
    chunk_count: int
    indexed_chunk_count: int


class QueryMetricsResponse(BaseModel):
    query_id: str
    retrieved_chunks: int
    context_tokens_sent: int
    response_tokens: int | None = None
