from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    model_id: str | None = None
    provider_name: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    metadata: dict[str, Any] | None = None


class ChatUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    model_id: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    is_active: bool | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    user_id: str
    model_id: str | None = None
    provider_name: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ConversationCreate(BaseModel):
    title: str = Field("New conversation", min_length=1, max_length=255)
    chat_id: str | None = None
    provider_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    rag_enabled: bool = False
    knowledge_base_id: str | None = None
    retrieval_config: dict[str, Any] | None = None


class ConversationUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=255)
    status: str | None = None
    provider_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    rag_enabled: bool | None = None
    knowledge_base_id: str | None = None
    retrieval_config: dict[str, Any] | None = None
    is_pinned: bool | None = None


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    chat_id: str | None = None
    user_id: str
    title: str
    status: str
    provider_name: str | None = None
    model_name: str | None = None
    model_id: str | None = None
    knowledge_base_id: str | None = None
    rag_enabled: bool
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    token_count: int = 0
    message_count: int = 0
    summary: str | None = None
    is_pinned: bool = False
    last_message_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse]
    total: int
    page: int
    page_size: int


class AttachmentInput(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., min_length=1, max_length=127)
    data_url: str


class MessageSend(BaseModel):
    content: str = Field("", max_length=200000)
    role: Literal["user"] = "user"
    parent_id: str | None = None
    attachment_ids: list[str] = Field(default_factory=list)
    attachments: list[AttachmentInput] = Field(default_factory=list)
    provider_name: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    stream: bool = True


class AttachmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    message_id: str
    kind: str
    file_name: str
    file_size: int
    mime_type: str
    storage_type: str
    created_at: datetime


class Citation(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    score: float
    content: str
    metadata: dict[str, Any] | None = None


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    role: str
    content: str
    provider_name: str | None = None
    model_name: str | None = None
    model_id: str | None = None
    provider_id: str | None = None
    tool_calls: list | None = None
    tool_call_id: str | None = None
    citations: list[dict[str, Any]] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    is_edited: bool = False
    created_at: datetime
    parent_id: str | None = None
    attachments: list[AttachmentResponse] = Field(default_factory=list)


class MessageListResponse(BaseModel):
    items: list[MessageResponse]
    total: int


class StreamChunk(BaseModel):
    content: str = ""
    done: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    provider: str | None = None


class StreamStatus(BaseModel):
    conversation_id: str
    active: bool
    request_id: str | None = None
