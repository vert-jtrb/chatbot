from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from sqlalchemy import CHAR, DateTime, String, TypeDecorator, event
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry


class GUID(TypeDecorator[str]):
    """Store UUID values as native ``uuid`` on PostgreSQL and CHAR(36) elsewhere.

    Python-facing values are always lowercase UUID strings, which keeps the
    service/API layer simple while satisfying native UUID storage in production.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return str(value)


class Base(DeclarativeBase):
    registry = registry(
        type_annotation_map={
            datetime: DateTime(timezone=True),
            str: String(255),
        }
    )


class Entity(Base):
    __abstract__ = True

    id: Mapped[str] = mapped_column(
        GUID(), primary_key=True, default=lambda: str(uuid4())
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


class SoftDeletable(Entity):
    __abstract__ = True

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None, nullable=True, index=True
    )
    is_deleted: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = ["Base", "Entity", "GUID", "SoftDeletable", "TimestampMixin", "utc_now"]
