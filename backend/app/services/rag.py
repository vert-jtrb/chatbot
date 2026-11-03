"""Retrieval augmented generation service.

The service owns knowledge-base lifecycle, document ingestion, hybrid
retrieval (dense vectors + keyword search), and citation generation.  It is
provider-agnostic: embeddings come from :class:`EmbeddingService`, while vector
persistence comes from the configured :mod:`app.services.vector_store` adapter.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import (
    Document,
    DocumentChunk,
    DocumentStatus,
    Embedding,
    KnowledgeBase,
)
from app.providers.ai.exceptions import ProviderError
from app.repositories.knowledge import (
    ChunkRepository,
    DocumentRepository,
    EmbeddingRepository,
    KnowledgeBaseRepository,
)
from app.services.base import ServiceResult
from app.services.documents import (
    DocumentExtractionError,
    chunk_text,
    detect_document_type,
    extract_document,
    keywords,
)
from app.services.embeddings import EmbeddingService
from app.services.storage import LocalStorage, StorageError
from app.services.vector_store import VectorRecord, get_vector_store

BATCH_SIZE = 24


class RAGService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.kb_repo = KnowledgeBaseRepository(session)
        self.doc_repo = DocumentRepository(session)
        self.chunk_repo = ChunkRepository(session)
        self.embedding_repo = EmbeddingRepository(session)
        self.storage = LocalStorage()

    # ------------------------------------------------------------------
    # Knowledge bases
    # ------------------------------------------------------------------
    async def create_knowledge_base(
        self, user_id: str, data: Any
    ) -> ServiceResult:
        name = getattr(data, "name", None) or str(data)
        provider = getattr(data, "embedding_provider", None) or settings.embedding_provider
        model = getattr(data, "embedding_model", None) or settings.resolved_embedding_model(
            provider
        )
        dimensions = settings.embedding_dimensions
        kb = await self.kb_repo.create(
            user_id=user_id,
            name=name,
            description=getattr(data, "description", None),
            embedding_provider=provider.lower(),
            embedding_model=model,
            embedding_dimensions=dimensions,
            chunk_size=getattr(data, "chunk_size", 1000),
            chunk_overlap=getattr(data, "chunk_overlap", 200),
            retrieval_config=getattr(data, "retrieval_config", None),
            document_count=0,
            is_active=True,
        )
        return ServiceResult.ok(kb, status_code=201)

    async def list_knowledge_bases(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> ServiceResult:
        bases, total = await self.kb_repo.get_by_user(user_id, skip, limit)
        return ServiceResult.ok(
            {
                "items": bases,
                "total": total,
                "page": skip // limit + 1 if limit else 1,
                "page_size": limit,
            }
        )

    async def get_knowledge_base(self, user_id: str, kb_id: str) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        return ServiceResult.ok(kb)

    async def update_knowledge_base(
        self, user_id: str, kb_id: str, data: Any
    ) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        payload = data.model_dump(exclude_unset=True) if hasattr(data, "model_dump") else dict(data)
        for key in ("embedding_provider", "embedding_model", "embedding_dimensions"):
            payload.pop(key, None)
        for key, value in payload.items():
            if hasattr(kb, key):
                setattr(kb, key, value)
        if payload:
            # Changing chunking settings requires re-indexing; mark it clearly.
            if "chunk_size" in payload or "chunk_overlap" in payload:
                kb.metadata_ = {
                    **(kb.metadata_ or {}),
                    "requires_reindex": True,
                }
        await self.session.flush()
        return ServiceResult.ok(kb)

    async def delete_knowledge_base(self, user_id: str, kb_id: str) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        documents, _ = await self.doc_repo.get_by_knowledge_base(kb_id, limit=10_000)
        for document in documents:
            await self._purge_document(document)
            document.is_deleted = True
            document.deleted_at = datetime.now(UTC)
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.knowledge_base_id == kb_id)
        )
        await self.session.execute(
            delete(Embedding).where(Embedding.knowledge_base_id == kb_id)
        )
        await self.kb_repo.delete(kb_id, soft=True)
        return ServiceResult.ok({"message": "Knowledge base deleted", "id": kb_id})

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    async def upload_document(
        self,
        user_id: str,
        kb_id: str,
        *,
        file_name: str,
        data: bytes,
        mime_type: str,
        process: bool = True,
    ) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        try:
            stored = self.storage.save_bytes(
                kind="documents",
                filename=file_name,
                data=data,
                mime_type=mime_type,
                allowed=settings.allowed_upload_extensions,
            )
        except StorageError as exc:
            return ServiceResult.error_response(str(exc), 400)
        document_type = detect_document_type(stored.file_name, mime_type)
        document = await self.doc_repo.create(
            knowledge_base_id=kb_id,
            user_id=user_id,
            file_name=stored.file_name,
            storage_path=stored.storage_path,
            file_size=stored.file_size,
            mime_type=stored.mime_type,
            document_type=document_type,
            status=DocumentStatus.PENDING,
            content_hash=stored.checksum,
        )
        await self.doc_repo.refresh_kb_count(kb_id)
        if process:
            result = await self.process_document(user_id, document.id)
            if not result.success:
                return ServiceResult.ok(document, status_code=201)
            await self.session.refresh(document)
        return ServiceResult.ok(document, status_code=201)

    async def list_documents(
        self, user_id: str, kb_id: str, skip: int = 0, limit: int = 50
    ) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        documents, total = await self.doc_repo.get_by_knowledge_base(kb_id, skip, limit)
        return ServiceResult.ok(
            {
                "items": documents,
                "total": total,
                "page": skip // limit + 1 if limit else 1,
                "page_size": limit,
            }
        )

    async def get_document(self, user_id: str, document_id: str) -> ServiceResult:
        document = await self.doc_repo.get_owned(document_id, user_id)
        if document is None:
            return ServiceResult.error_response("Document not found", 404)
        return ServiceResult.ok(document)

    async def process_document(self, user_id: str, document_id: str) -> ServiceResult:
        document = await self.doc_repo.get_owned(document_id, user_id)
        if document is None:
            return ServiceResult.error_response("Document not found", 404)
        kb = await self.kb_repo.get_owned(document.knowledge_base_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)

        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        await self.session.flush()

        try:
            extracted = await asyncio.to_thread(
                extract_document, Path(document.storage_path), document.file_name
            )
            pieces = chunk_text(
                extracted.text,
                chunk_size=kb.chunk_size,
                chunk_overlap=kb.chunk_overlap,
            )
            if not pieces:
                raise DocumentExtractionError(
                    "No readable text was found in the document"
                )

            await self._delete_document_chunks(document.id)
            vectors = await self._embed_in_batches(
                pieces,
                provider=kb.embedding_provider,
                model=kb.embedding_model,
                dimensions=kb.embedding_dimensions,
            )

            chunks: list[DocumentChunk] = []
            for index, content in enumerate(pieces):
                chunk = DocumentChunk(
                    document_id=document.id,
                    knowledge_base_id=kb.id,
                    content=content,
                    chunk_index=index,
                    token_count=max(1, len(content) // 4),
                    metadata_={
                        "document_name": document.file_name,
                        "document_type": str(document.document_type),
                    },
                )
                self.session.add(chunk)
                chunks.append(chunk)
            await self.session.flush()

            store = get_vector_store(self.session, kb)
            try:
                await store.upsert(
                    [
                        VectorRecord(
                            chunk_id=chunk.id,
                            vector=vector,
                            content=chunk.content,
                            metadata={
                                "document_id": document.id,
                                "document_name": document.file_name,
                                "chunk_index": chunk.chunk_index,
                                "knowledge_base_id": kb.id,
                            },
                        )
                        for chunk, vector in zip(chunks, vectors, strict=False)
                    ]
                )
            finally:
                await store.aclose()

            document.status = DocumentStatus.READY
            document.chunk_count = len(chunks)
            document.page_count = extracted.page_count
            document.processed_at = datetime.now(UTC)
            document.metadata_ = {
                **(document.metadata_ or {}),
                "characters": len(extracted.text),
                "embedding_provider": kb.embedding_provider,
                "embedding_model": kb.embedding_model,
            }
            await self.session.flush()
            await self.doc_repo.refresh_kb_count(kb.id)
            return ServiceResult.ok(
                {
                    "document_id": document.id,
                    "status": str(document.status),
                    "chunks": len(chunks),
                    "page_count": document.page_count,
                }
            )
        except (DocumentExtractionError, StorageError, ProviderError) as exc:
            await self._mark_failed(document, str(exc))
            return ServiceResult.error_response(f"Ingestion failed: {exc}", 400)
        except Exception as exc:  # pragma: no cover - defensive
            await self._mark_failed(document, str(exc))
            return ServiceResult.error_response(f"Ingestion failed: {exc}", 500)

    async def reindex_document(self, user_id: str, document_id: str) -> ServiceResult:
        return await self.process_document(user_id, document_id)

    async def reindex_knowledge_base(
        self, user_id: str, kb_id: str
    ) -> ServiceResult:
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        documents, _ = await self.doc_repo.get_by_knowledge_base(kb_id, limit=10_000)
        processed = 0
        chunks = 0
        failures: list[dict[str, str]] = []
        for document in documents:
            result = await self.process_document(user_id, document.id)
            if result.success:
                processed += 1
                chunks += int(result.data.get("chunks") or 0)
            else:
                failures.append(
                    {"document_id": document.id, "error": result.error or "unknown"}
                )
        await self.doc_repo.refresh_kb_count(kb_id)
        return ServiceResult.ok(
            {
                "knowledge_base_id": kb_id,
                "documents": processed,
                "chunks": chunks,
                "embeddings": chunks,
                "failures": failures,
            }
        )

    async def delete_document(self, user_id: str, document_id: str) -> ServiceResult:
        document = await self.doc_repo.get_owned(document_id, user_id)
        if document is None:
            return ServiceResult.error_response("Document not found", 404)
        kb_id = document.knowledge_base_id
        await self._purge_document(document)
        document.is_deleted = True
        document.deleted_at = datetime.now(UTC)
        document.status = DocumentStatus.DELETED
        await self.session.flush()
        await self.doc_repo.refresh_kb_count(kb_id)
        return ServiceResult.ok({"message": "Document deleted", "id": document_id})

    async def _purge_document(self, document: Document) -> None:
        chunk_ids = (
            (
                await self.session.execute(
                    select(DocumentChunk.id).where(
                        DocumentChunk.document_id == document.id
                    )
                )
            )
            .scalars()
            .all()
        )
        kb = await self.session.get(KnowledgeBase, document.knowledge_base_id)
        if kb is not None and chunk_ids:
            store = get_vector_store(self.session, kb)
            try:
                await store.delete(chunk_ids)
            except Exception:
                # The database rows below are still authoritative.
                pass
            finally:
                await store.aclose()
        await self._delete_document_chunks(document.id)
        self.storage.delete(document.storage_path)

    async def _delete_document_chunks(self, document_id: str) -> None:
        chunk_ids = (
            await self.session.execute(
                select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
            )
        ).scalars().all()
        if chunk_ids:
            await self.session.execute(
                delete(Embedding).where(Embedding.chunk_id.in_(list(chunk_ids)))
            )
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        await self.session.flush()

    async def _mark_failed(self, document: Document, error: str) -> None:
        document.status = DocumentStatus.ERROR
        document.error_message = error[:2000]
        document.processed_at = datetime.now(UTC)
        await self.session.flush()

    async def _embed_in_batches(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        service = EmbeddingService(self.session)
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), BATCH_SIZE):
            batch = texts[offset : offset + BATCH_SIZE]
            result = await service.embed(
                batch,
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
            vectors.extend(result.vectors)
        if len(vectors) != len(texts):
            raise ProviderError("Embedding provider returned an incomplete batch")
        return vectors

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def search(
        self,
        user_id: str,
        *,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        hybrid_alpha: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> ServiceResult:
        started = time.perf_counter()
        kb = await self.kb_repo.get_owned(kb_id, user_id)
        if kb is None:
            return ServiceResult.error_response("Knowledge base not found", 404)
        top_k = top_k or settings.rag_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.rag_similarity_threshold
        )
        alpha = hybrid_alpha if hybrid_alpha is not None else settings.rag_hybrid_alpha
        terms = keywords(query)

        try:
            embedding = await EmbeddingService(self.session).embed(
                [query],
                provider=kb.embedding_provider,
                model=kb.embedding_model,
                dimensions=kb.embedding_dimensions,
            )
            query_vector = embedding.vectors[0]
        except ProviderError as exc:
            return ServiceResult.error_response(
                f"Could not create a query embedding: {exc}", 502
            )

        store = get_vector_store(self.session, kb)
        try:
            matches = await store.search(
                query_vector,
                limit=max(top_k * 4, top_k),
                metadata_filter=None,
            )
        except ProviderError as exc:
            return ServiceResult.error_response(f"Vector search failed: {exc}", 502)
        finally:
            await store.aclose()

        vector_scores = {match.chunk_id: match.score for match in matches}
        keyword_chunks = await self.chunk_repo.keyword_search(kb_id, terms, limit=top_k * 6)
        keyword_scores = {
            chunk.id: _keyword_score(query, chunk.content) for chunk in keyword_chunks
        }
        candidate_ids = set(vector_scores) | set(keyword_scores)
        if not candidate_ids:
            return ServiceResult.ok(
                {
                    "chunks": [],
                    "query": query,
                    "total_chunks": 0,
                    "knowledge_base_id": kb_id,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                }
            )

        rows = (
            await self.session.execute(
                select(DocumentChunk, Document)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(
                    DocumentChunk.id.in_(list(candidate_ids)),
                    Document.is_deleted.is_(False),
                )
            )
        ).all()
        results: list[dict[str, Any]] = []
        for chunk, document in rows:
            if not _matches_filter(chunk, document, metadata_filter):
                continue
            vector_score = float(vector_scores.get(chunk.id, 0.0))
            keyword_score = float(keyword_scores.get(chunk.id, 0.0))
            score = (alpha * vector_score) + ((1.0 - alpha) * keyword_score)
            if vector_score < threshold and keyword_score <= 0:
                continue
            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "document_name": document.file_name,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "score": round(score, 6),
                    "vector_score": round(vector_score, 6),
                    "keyword_score": round(keyword_score, 6),
                    "metadata": {
                        **(document.metadata_ or {}),
                        **(chunk.metadata_ or {}),
                    },
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:top_k]
        return ServiceResult.ok(
            {
                "chunks": results,
                "query": query,
                "total_chunks": len(results),
                "knowledge_base_id": kb_id,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )

    async def build_context(
        self,
        user_id: str,
        *,
        kb_id: str,
        query: str,
        config: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], ServiceResult | None]:
        config = config or {}
        result = await self.search(
            user_id,
            kb_id=kb_id,
            query=query,
            top_k=int(config.get("top_k") or settings.rag_top_k),
            similarity_threshold=config.get("similarity_threshold"),
            hybrid_alpha=config.get("hybrid_alpha"),
            metadata_filter=config.get("metadata_filter"),
        )
        if not result.success:
            return "", [], result
        chunks: Sequence[dict[str, Any]] = result.data["chunks"]
        budget = settings.rag_max_context_chars
        blocks: list[str] = []
        citations: list[dict[str, Any]] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            content = str(chunk["content"])
            if used + len(content) > budget:
                content = content[: max(0, budget - used)]
            if not content:
                break
            used += len(content)
            blocks.append(
                f"[{index}] Source: {chunk['document_name']} "
                f"(chunk {chunk['chunk_index']})\n{content}"
            )
            citations.append(
                {
                    "document_id": chunk["document_id"],
                    "document_name": chunk["document_name"],
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk["chunk_index"],
                    "score": chunk["score"],
                    "content": content[:1200],
                    "metadata": chunk.get("metadata"),
                }
            )
            if used >= budget:
                break
        if not blocks:
            return "", [], None
        context = (
            "Use the following knowledge-base excerpts to answer the user's "
            "question. Cite sources using their [n] reference when relevant. "
            "If the answer is not present, say so.\n\n" + "\n\n".join(blocks)
        )
        return context, citations, None


def _keyword_score(query: str, content: str) -> float:
    terms = keywords(query)
    if not terms:
        return 0.0
    lowered = content.lower()
    hits = sum(1 for term in terms if term in lowered)
    score = hits / len(terms)
    if query.strip().lower() in lowered:
        score = max(score, 0.95)
    return round(min(1.0, score), 6)


def _matches_filter(
    chunk: DocumentChunk,
    document: Document,
    metadata_filter: dict[str, Any] | None,
) -> bool:
    if not metadata_filter:
        return True
    merged = {**(document.metadata_ or {}), **(chunk.metadata_ or {})}
    for key, expected in metadata_filter.items():
        if key not in merged:
            return False
        actual = merged[key]
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


__all__ = ["RAGService"]
