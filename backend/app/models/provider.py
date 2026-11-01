from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import GUID, Entity


class ProviderType(StrEnum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    LM_STUDIO = "lm_studio"
    VLLM = "vllm"
    CUSTOM = "custom"


class AIProviderConfig(Entity):
    """Runtime provider configuration.

    API keys are always stored encrypted (Fernet) and never returned by any API
    schema.  Multiple instances of the same provider type are supported, e.g.
    ``huggingface`` and ``hf-dedicated`` for a dedicated Inference Endpoint.
    """

    __tablename__ = "ai_provider_configurations"

    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider_type: Mapped[ProviderType] = mapped_column(String(50), nullable=False)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    default_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    default_inference_provider: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    timeout: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_retries: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    health_status: Mapped[str] = mapped_column(
        String(20), default="unknown", nullable=False
    )
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    models: Mapped[list[AIModel]] = relationship(
        back_populates="provider", cascade="all, delete-orphan"
    )


class AIModel(Entity):
    __tablename__ = "ai_models"
    __table_args__ = (
        UniqueConstraint("provider_id", "model_id", name="uq_model_per_provider"),
    )

    provider_id: Mapped[str] = mapped_column(
        GUID(),
        ForeignKey("ai_provider_configurations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task: Mapped[str | None] = mapped_column(String(100), nullable=True)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supports_embeddings: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    supports_audio: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    supports_json_mode: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    input_price_per_million: Mapped[float | None] = mapped_column(Float, nullable=True)
    output_price_per_million: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    provider: Mapped[AIProviderConfig] = relationship(back_populates="models")


# Backwards-compatible name used by older service modules.
AIProvider = AIProviderConfig

__all__ = ["AIProvider", "AIProviderConfig", "AIModel", "ProviderType"]
