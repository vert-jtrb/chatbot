from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


def _engine_kwargs() -> dict:
    kwargs: dict = {"echo": settings.database_echo or settings.debug, "pool_pre_ping": True}
    if not settings.is_sqlite:
        kwargs.update(
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_recycle=settings.database_pool_recycle,
        )
    return kwargs


engine = create_async_engine(str(settings.database_url), **_engine_kwargs())

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Request-scoped session that commits once the handler succeeds.

    Services call ``flush`` so they can observe generated ids; the dependency
    owns the transaction boundary.  Any unhandled exception rolls the whole
    request back, which keeps multi-row writes atomic.
    """

    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


async def session_scope() -> AsyncGenerator[AsyncSession]:
    """Commit-on-success session for background jobs and scripts."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
