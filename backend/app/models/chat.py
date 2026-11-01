from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import GUID, Entity, SoftDeletable

if TYPE_CHECKING:
    from app.models.knowledge import KnowledgeBase
    from app.models.provider import AIModel, AIProviderConfig
    from app.models.user import User


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class AttachmentKind(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"


class Chat(Entity):
    """Optional reusable chat template.

    Conversations can be created directly (``chat_id`` is nullable), while the
    ``/chats`` endpoints remain available for users who want reusable presets.
    """

    __tablename__ = "chats"

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_id: Mapped[str | None] = mapped_column(
        GUID(), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, default=0.7)
    top_p: Mapped[float | None] = mapped_column(Float, default=1.0)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    user: Mapped[User] = relationship("User")
    conversations: Mapped[list[Conversation]] = relationship(
        "Conversation", back_populates="chat", cascade="all, delete-orphan"
    )


class Conversation(SoftDeletable):
    __tablename__ = "conversations"

    chat_id: Mapped[str | None] = mapped_column(
        GUID(),
        ForeignKey("chats.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ConversationStatus] = mapped_column(
        String(20), default=ConversationStatus.ACTIVE, nullable=False, index=True
    )
    provider_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_id: Mapped[str | None] = mapped_column(
        GUID(), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    knowledge_base_id: Mapped[str | None] = mapped_column(
        GUID(),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    rag_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    retrieval_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, default=0.7)
    top_p: Mapped[float | None] = mapped_column(Float, default=1.0)
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    chat: Mapped[Chat | None] = relationship("Chat", back_populates="conversations")
    user: Mapped[User] = relationship("User", back_populates="conversations")
    model: Mapped[AIModel | None] = relationship(
        "AIModel", foreign_keys=[model_id], lazy="joined"
    )
    knowledge_base: Mapped[KnowledgeBase | None] = relationship(
        "KnowledgeBase", foreign_keys=[knowledge_base_id]
    )
    messages: Mapped[list[Message]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    @property
    def effective_provider(self) -> str | None:
        if self.provider_name:
            return self.provider_name
        return self.model.provider.name if self.model else None

    @property
    def effective_model(self) -> str | None:
        if self.model_name:
            return self.model_name
        return self.model.model_id if self.model else None


class Message(Entity):
    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(
        GUID(),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    provider_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_id: Mapped[str | None] = mapped_column(
        GUID(), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    provider_id: Mapped[str | None] = mapped_column(
        GUID(),
        ForeignKey("ai_provider_configurations.id", ondelete="SET NULL"),
        nullable=True,
    )
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finish_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        GUID(), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )

    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="messages"
    )
    model: Mapped[AIModel | None] = relationship(
        "AIModel", foreign_keys=[model_id], lazy="joined"
    )
    provider: Mapped[AIProviderConfig | None] = relationship(
        "AIProviderConfig", foreign_keys=[provider_id]
    )
    attachments: Mapped[list[MessageAttachment]] = relationship(
        "MessageAttachment",
        back_populates="message",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class MessageAttachment(Entity):
    __tablename__ = "message_attachments"

    message_id: Mapped[str] = mapped_column(
        GUID(),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[AttachmentKind] = mapped_column(
        String(20), default=AttachmentKind.FILE, nullable=False
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    storage_type: Mapped[str] = mapped_column(String(50), default="local")
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    message: Mapped[Message] = relationship(
        "Message", back_populates="attachments"
    )


__all__ = [
    "AttachmentKind",
    "Chat",
    "Conversation",
    "ConversationStatus",
    "Message",
    "MessageAttachment",
    "MessageRole",
]
