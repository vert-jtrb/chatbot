from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    embedding_provider: str = "huggingface"
    embedding_model: str | None = None
    chunk_size: int = Field(1000, ge=100, le=8000)
    chunk_overlap: int = Field(200, ge=0, le=2000)
    retrieval_config: dict[str, Any] | None = None


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    chunk_size: int | None = Field(None, ge=100, le=8000)
    chunk_overlap: int | None = Field(None, ge=0, le=2000)
    retrieval_config: dict[str, Any] | None = None
    is_active: bool | None = None


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str | None = None
    user_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    document_count: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class KnowledgeBaseListResponse(BaseModel):
    items: list[KnowledgeBaseResponse]
    total: int
    page: int
    page_size: int


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    knowledge_base_id: str
    file_name: str
    file_size: int
    mime_type: str
    document_type: str
    status: str
    chunk_count: int
    error_message: str | None = None
    created_at: datetime


class DocumentResponse(DocumentUploadResponse):
    content_hash: str | None = None
    page_count: int | None = None
    processed_at: datetime | None = None
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    knowledge_base_id: str
    top_k: int = Field(6, ge=1, le=50)
    similarity_threshold: float = Field(0.25, ge=0.0, le=1.0)
    hybrid_alpha: float = Field(0.65, ge=0.0, le=1.0)
    metadata_filter: dict[str, Any] | None = None


class RAGChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    chunk_index: int
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    metadata: dict[str, Any] | None = None


class RAGResult(BaseModel):
    chunks: list[RAGChunk]
    query: str
    total_chunks: int
    knowledge_base_id: str
    latency_ms: int


class ReindexResponse(BaseModel):
    knowledge_base_id: str
    documents: int
    chunks: int
    embeddings: int


class DeleteResponse(BaseModel):
    message: str
    id: str


class IngestionStatus(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    error: str | None = None
