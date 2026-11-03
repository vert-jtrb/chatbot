"""API serializers.

SQLAlchemy models are never returned directly: enums are normalised, secret
fields are excluded, and computed properties are resolved while the session is
still open.
"""

from __future__ import annotations

from typing import Any

from app.models.chat import Conversation, Message, MessageAttachment
from app.models.user import User


def user_payload(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "email_verified": user.email_verified,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "role": user.role_name,
        "two_factor_enabled": user.two_factor_enabled,
        "last_login_at": user.last_login_at,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
    }


def auth_payload(user: User, tokens: dict[str, Any]) -> dict[str, Any]:
    return {"user": user_payload(user), **tokens}


def attachment_payload(attachment: MessageAttachment) -> dict[str, Any]:
    return {
        "id": attachment.id,
        "message_id": attachment.message_id,
        "kind": str(attachment.kind),
        "file_name": attachment.file_name,
        "file_size": attachment.file_size,
        "mime_type": attachment.mime_type,
        "storage_type": attachment.storage_type,
        "created_at": attachment.created_at,
    }


def message_payload(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "conversation_id": message.conversation_id,
        "role": str(message.role),
        "content": message.content,
        "provider_name": message.provider_name,
        "model_name": message.model_name,
        "model_id": message.model_id,
        "provider_id": message.provider_id,
        "tool_calls": message.tool_calls,
        "tool_call_id": message.tool_call_id,
        "citations": message.citations,
        "input_tokens": message.input_tokens,
        "output_tokens": message.output_tokens,
        "total_tokens": message.total_tokens,
        "latency_ms": message.latency_ms,
        "finish_reason": message.finish_reason,
        "is_edited": message.is_edited,
        "created_at": message.created_at,
        "parent_id": message.parent_id,
        "attachments": [
            attachment_payload(attachment)
            for attachment in getattr(message, "attachments", []) or []
        ],
    }


def conversation_payload(conversation: Conversation) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "chat_id": conversation.chat_id,
        "user_id": conversation.user_id,
        "title": conversation.title,
        "status": str(conversation.status),
        "provider_name": conversation.effective_provider,
        "model_name": conversation.effective_model,
        "model_id": conversation.model_id,
        "knowledge_base_id": conversation.knowledge_base_id,
        "rag_enabled": conversation.rag_enabled,
        "system_prompt": conversation.system_prompt,
        "temperature": conversation.temperature,
        "top_p": conversation.top_p,
        "max_tokens": conversation.max_tokens,
        "token_count": conversation.token_count,
        "message_count": conversation.message_count,
        "summary": conversation.summary,
        "is_pinned": conversation.is_pinned,
        "last_message_at": conversation.last_message_at,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


__all__ = [
    "attachment_payload",
    "auth_payload",
    "conversation_payload",
    "message_payload",
    "user_payload",
]
