from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import GUID, Entity


class VectorIndex(Entity):
    """External vector-store bookkeeping for non-pgvector backends."""

    __tablename__ = "vector_indices"

    document_chunk_id: Mapped[str | None] = mapped_column(
        GUID(),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    vector_store_type: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)


__all__ = ["VectorIndex"]
