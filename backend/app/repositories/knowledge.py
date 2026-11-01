from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.knowledge import Document, DocumentChunk, Embedding, KnowledgeBase
from app.repositories.base import BaseRepository


class KnowledgeBaseRepository(BaseRepository[KnowledgeBase]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, KnowledgeBase)

    async def get_owned(self, kb_id: str, user_id: str) -> KnowledgeBase | None:
        result = await self.session.execute(
            select(KnowledgeBase).where(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.is_deleted.is_(False),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> tuple[Sequence[KnowledgeBase], int]:
        where = (
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.is_deleted.is_(False),
        )
        total = await self.session.scalar(
            select(func.count()).select_from(KnowledgeBase).where(*where)
        )
        result = await self.session.execute(
            select(KnowledgeBase)
            .where(*where)
            .order_by(KnowledgeBase.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)


class DocumentRepository(BaseRepository[Document]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Document)

    async def get_owned(self, document_id: str, user_id: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.is_deleted.is_(False),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_knowledge_base(
        self, kb_id: str, skip: int = 0, limit: int = 50
    ) -> tuple[Sequence[Document], int]:
        where = (
            Document.knowledge_base_id == kb_id,
            Document.is_deleted.is_(False),
        )
        total = await self.session.scalar(
            select(func.count()).select_from(Document).where(*where)
        )
        result = await self.session.execute(
            select(Document)
            .where(*where)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)

    async def get_with_chunks(self, document_id: str) -> Document | None:
        result = await self.session.execute(
            select(Document)
            .options(joinedload(Document.chunks))
            .where(Document.id == document_id)
        )
        return result.unique().scalar_one_or_none()

    async def refresh_kb_count(self, kb_id: str) -> None:
        total = await self.session.scalar(
            select(func.count())
            .select_from(Document)
            .where(
                Document.knowledge_base_id == kb_id,
                Document.is_deleted.is_(False),
            )
        )
        kb = await self.session.get(KnowledgeBase, kb_id)
        if kb is not None:
            kb.document_count = int(total or 0)
            await self.session.flush()


class ChunkRepository(BaseRepository[DocumentChunk]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, DocumentChunk)

    async def delete_for_document(self, document_id: str) -> None:
        await self.session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        await self.session.flush()

    async def list_for_kb(self, kb_id: str) -> Sequence[DocumentChunk]:
        result = await self.session.execute(
            select(DocumentChunk).where(DocumentChunk.knowledge_base_id == kb_id)
        )
        return result.scalars().all()

    async def keyword_search(
        self, kb_id: str, terms: list[str], limit: int = 30
    ) -> Sequence[DocumentChunk]:
        if not terms:
            return []
        clauses = [DocumentChunk.content.ilike(f"%{term}%") for term in terms[:8]]
        result = await self.session.execute(
            select(DocumentChunk)
            .where(
                DocumentChunk.knowledge_base_id == kb_id,
                or_(*clauses),
            )
            .limit(limit)
        )
        return result.scalars().all()


class EmbeddingRepository(BaseRepository[Embedding]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Embedding)

    async def delete_for_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        await self.session.execute(
            delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids))
        )
        await self.session.flush()

    async def list_for_kb(self, kb_id: str) -> Sequence[Embedding]:
        result = await self.session.execute(
            select(Embedding).where(Embedding.knowledge_base_id == kb_id)
        )
        return result.scalars().all()


__all__ = [
    "ChunkRepository",
    "DocumentRepository",
    "EmbeddingRepository",
    "KnowledgeBaseRepository",
]
