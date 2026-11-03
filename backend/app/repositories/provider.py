from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.provider import AIModel, AIProvider, AIProviderConfig
from app.repositories.base import BaseRepository


class AIProviderRepository(BaseRepository[AIProvider]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AIProvider)

    async def get_with_models(self, provider_id: str) -> AIProvider | None:
        stmt = (
            select(AIProvider)
            .options(joinedload(AIProvider.models))
            .where(AIProvider.id == provider_id)
        )
        result = await self.session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def get_by_name(
        self, name: str, *, include_deleted: bool = False
    ) -> AIProvider | None:
        stmt = select(AIProvider).where(AIProvider.name == name)
        if not include_deleted:
            stmt = stmt.where(AIProvider.is_deleted.is_(False))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_default(self) -> AIProvider | None:
        result = await self.session.execute(
            select(AIProvider).where(
                AIProvider.is_default.is_(True), AIProvider.is_active.is_(True)
            )
        )
        return result.scalars().first()

    async def get_active(self) -> Sequence[AIProvider]:
        result = await self.session.execute(
            select(AIProvider)
            .where(AIProvider.is_active.is_(True))
            .order_by(AIProvider.is_default.desc(), AIProvider.name)
        )
        return result.scalars().all()

    async def clear_default(self, *, except_id: str | None = None) -> None:
        stmt = update(AIProvider).values(is_default=False)
        if except_id:
            stmt = stmt.where(AIProvider.id != except_id)
        await self.session.execute(stmt)
        await self.session.flush()


class AIModelRepository(BaseRepository[AIModel]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AIModel)

    async def get_by_provider(self, provider_id: str) -> Sequence[AIModel]:
        result = await self.session.execute(
            select(AIModel)
            .where(AIModel.provider_id == provider_id, AIModel.is_active.is_(True))
            .order_by(AIModel.name)
        )
        return result.scalars().all()

    async def get_default_by_provider(self, provider_id: str) -> AIModel | None:
        result = await self.session.execute(
            select(AIModel).where(
                AIModel.provider_id == provider_id,
                AIModel.is_default.is_(True),
                AIModel.is_active.is_(True),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_model_id(
        self, model_id: str, provider_id: str | None = None
    ) -> AIModel | None:
        stmt = select(AIModel).where(
            AIModel.model_id == model_id, AIModel.is_active.is_(True)
        )
        if provider_id:
            stmt = stmt.where(AIModel.provider_id == provider_id)
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_with_provider(self, model_id: str) -> AIModel | None:
        result = await self.session.execute(
            select(AIModel)
            .options(joinedload(AIModel.provider))
            .where(AIModel.id == model_id, AIModel.is_active.is_(True))
        )
        return result.unique().scalar_one_or_none()

    async def list_available(
        self,
        *,
        provider_type: str | None = None,
        supports_vision: bool | None = None,
        supports_tools: bool | None = None,
        supports_embeddings: bool | None = None,
        search: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[AIModel], int]:
        stmt = (
            select(AIModel)
            .join(AIProvider, AIModel.provider_id == AIProvider.id)
            .where(
                AIModel.is_active.is_(True),
                AIProvider.is_active.is_(True),
                AIProvider.is_deleted.is_(False),
            )
        )
        if provider_type:
            stmt = stmt.where(AIProvider.provider_type == provider_type)
        if supports_vision is not None:
            stmt = stmt.where(AIModel.supports_vision.is_(supports_vision))
        if supports_tools is not None:
            stmt = stmt.where(AIModel.supports_tools.is_(supports_tools))
        if supports_embeddings is not None:
            stmt = stmt.where(AIModel.supports_embeddings.is_(supports_embeddings))
        if search:
            stmt = stmt.where(AIModel.name.ilike(f"%{search}%"))
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await self.session.scalar(count_stmt)
        result = await self.session.execute(
            stmt.order_by(AIProvider.is_default.desc(), AIModel.name)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)

    async def upsert(self, provider_id: str, payload: dict) -> AIModel:
        existing = await self.session.execute(
            select(AIModel).where(
                AIModel.provider_id == provider_id,
                AIModel.model_id == payload["model_id"],
            )
        )
        model = existing.scalar_one_or_none()
        if model is None:
            return await self.create(provider_id=provider_id, **payload)
        for key, value in payload.items():
            if value is not None:
                setattr(model, key, value)
        await self.session.flush()
        return model


__all__ = ["AIModelRepository", "AIProviderRepository"]
