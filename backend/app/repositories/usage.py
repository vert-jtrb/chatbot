from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog
from app.models.usage import UsageRecord
from app.repositories.base import BaseRepository


class UsageRepository(BaseRepository[UsageRecord]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, UsageRecord)

    async def summary(
        self,
        *,
        user_id: str | None = None,
        since: datetime | None = None,
        provider: str | None = None,
    ) -> dict:
        where = []
        if user_id:
            where.append(UsageRecord.user_id == user_id)
        if since:
            where.append(UsageRecord.created_at >= since)
        if provider:
            where.append(UsageRecord.provider == provider)
        stmt = select(
            func.count(UsageRecord.id),
            func.coalesce(func.sum(UsageRecord.input_tokens), 0),
            func.coalesce(func.sum(UsageRecord.output_tokens), 0),
            func.coalesce(func.sum(UsageRecord.total_tokens), 0),
            func.coalesce(func.sum(UsageRecord.cost_usd), 0.0),
            func.coalesce(func.avg(UsageRecord.latency_ms), 0.0),
        ).where(*where)
        row = (await self.session.execute(stmt)).one()
        return {
            "requests": int(row[0] or 0),
            "input_tokens": int(row[1] or 0),
            "output_tokens": int(row[2] or 0),
            "total_tokens": int(row[3] or 0),
            "cost_usd": round(float(row[4] or 0.0), 6),
            "average_latency_ms": round(float(row[5] or 0.0), 2),
        }

    async def by_provider(
        self, *, user_id: str | None = None, since: datetime | None = None
    ) -> Sequence[dict]:
        where = []
        if user_id:
            where.append(UsageRecord.user_id == user_id)
        if since:
            where.append(UsageRecord.created_at >= since)
        stmt = (
            select(
                UsageRecord.provider,
                UsageRecord.model,
                func.count(UsageRecord.id),
                func.coalesce(func.sum(UsageRecord.total_tokens), 0),
                func.coalesce(func.sum(UsageRecord.cost_usd), 0.0),
            )
            .where(*where)
            .group_by(UsageRecord.provider, UsageRecord.model)
            .order_by(func.count(UsageRecord.id).desc())
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "provider": row[0],
                "model": row[1],
                "requests": int(row[2]),
                "total_tokens": int(row[3] or 0),
                "cost_usd": round(float(row[4] or 0.0), 6),
            }
            for row in rows
        ]


class AuditRepository(BaseRepository[AuditLog]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, AuditLog)

    async def list_logs(
        self,
        *,
        user_id: str | None = None,
        action: str | None = None,
        severity: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> tuple[Sequence[AuditLog], int]:
        where = []
        if user_id:
            where.append(AuditLog.user_id == user_id)
        if action:
            where.append(AuditLog.action == action)
        if severity:
            where.append(AuditLog.severity == severity)
        total = await self.session.scalar(
            select(func.count()).select_from(AuditLog).where(*where)
        )
        result = await self.session.execute(
            select(AuditLog)
            .where(*where)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)


__all__ = ["AuditRepository", "UsageRepository"]
