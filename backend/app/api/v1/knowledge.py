"""Knowledge-base, document ingestion, and retrieval endpoints."""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.pagination import pagination_params
from app.core.config import settings
from app.database.session import get_session
from app.models.user import User
from app.schemas.rag import (
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IngestionStatus,
    KnowledgeBaseCreate,
    KnowledgeBaseListResponse,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    RAGQuery,
    RAGResult,
    ReindexResponse,
)
from app.services.rag import RAGService

router = APIRouter(prefix="/knowledge", tags=["Knowledge Bases"])
logger = structlog.get_logger()


def _raise(result: Any) -> Any:
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


def _kb_payload(kb: Any) -> dict[str, Any]:
    return {
        "id": kb.id,
        "name": kb.name,
        "description": kb.description,
        "user_id": kb.user_id,
        "embedding_provider": kb.embedding_provider,
        "embedding_model": kb.embedding_model,
        "embedding_dimensions": kb.embedding_dimensions,
        "chunk_size": kb.chunk_size,
        "chunk_overlap": kb.chunk_overlap,
        "document_count": kb.document_count,
        "is_active": kb.is_active,
        "created_at": kb.created_at,
        "updated_at": kb.updated_at,
    }


def _document_payload(document: Any, *, full: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": document.id,
        "knowledge_base_id": document.knowledge_base_id,
        "file_name": document.file_name,
        "file_size": document.file_size,
        "mime_type": document.mime_type,
        "document_type": str(document.document_type),
        "status": str(document.status),
        "chunk_count": document.chunk_count,
        "error_message": document.error_message,
        "created_at": document.created_at,
    }
    if full:
        payload.update(
            {
                "content_hash": document.content_hash,
                "page_count": document.page_count,
                "processed_at": document.processed_at,
                "updated_at": document.updated_at,
            }
        )
    return payload


# ----------------------------------------------------------------------
# Knowledge bases
# ----------------------------------------------------------------------
@router.post("/bases", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).create_knowledge_base(
        current_user.id, payload
    )
    return _kb_payload(_raise(result))


@router.get("/bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    params: dict = Depends(pagination_params),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).list_knowledge_bases(
        current_user.id, params["skip"], params["limit"]
    )
    data = _raise(result)
    return {
        "items": [_kb_payload(item) for item in data["items"]],
        "total": data["total"],
        "page": data["page"],
        "page_size": data["page_size"],
    }


@router.get("/bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).get_knowledge_base(current_user.id, kb_id)
    return _kb_payload(_raise(result))


@router.patch("/bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    payload: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).update_knowledge_base(
        current_user.id, kb_id, payload
    )
    return _kb_payload(_raise(result))


@router.delete("/bases/{kb_id}", response_model=DeleteResponse)
async def delete_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).delete_knowledge_base(
        current_user.id, kb_id
    )
    return _raise(result)


@router.post("/bases/{kb_id}/reindex", response_model=ReindexResponse)
async def reindex_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).reindex_knowledge_base(
        current_user.id, kb_id
    )
    return _raise(result)


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------
@router.post(
    "/bases/{kb_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=201,
)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    process: bool = Query(True, description="Index the document immediately"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_size_mb} MB limit",
        )
    result = await RAGService(session).upload_document(
        current_user.id,
        kb_id,
        file_name=file.filename or "upload",
        data=data,
        mime_type=file.content_type or "application/octet-stream",
        process=process,
    )
    return _document_payload(_raise(result))


@router.get(
    "/bases/{kb_id}/documents", response_model=DocumentListResponse
)
async def list_documents(
    kb_id: str,
    params: dict = Depends(pagination_params),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).list_documents(
        current_user.id,
        kb_id,
        params["skip"],
        params["limit"],
    )
    data = _raise(result)
    return {
        "items": [
            _document_payload(item, full=True) for item in data["items"]
        ],
        "total": data["total"],
        "page": data["page"],
        "page_size": data["page_size"],
    }


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).get_document(current_user.id, document_id)
    return _document_payload(_raise(result), full=True)


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).delete_document(
        current_user.id, document_id
    )
    return _raise(result)


@router.post(
    "/documents/{document_id}/reindex", response_model=IngestionStatus
)
async def reindex_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = RAGService(session)
    result = await service.reindex_document(current_user.id, document_id)
    data = _raise(result)
    return {
        "document_id": data.get("document_id", document_id),
        "status": data.get("status", "ready"),
        "chunk_count": int(data.get("chunks") or 0),
        "error": None,
    }


# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
@router.post("/search", response_model=RAGResult)
async def search_knowledge(
    payload: RAGQuery,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await RAGService(session).search(
        current_user.id,
        kb_id=payload.knowledge_base_id,
        query=payload.query,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
        hybrid_alpha=payload.hybrid_alpha,
        metadata_filter=payload.metadata_filter,
    )
    return _raise(result)
