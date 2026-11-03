"""Chat and conversation endpoints, including SSE streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.pagination import pagination_params
from app.api.v1.serializers import (
    conversation_payload,
    message_payload,
)
from app.database.session import async_session_factory, get_session
from app.models.chat import Conversation, Message, MessageRole
from app.models.user import User
from app.providers.ai.exceptions import ProviderError
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatUpdate,
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageListResponse,
    MessageResponse,
    MessageSend,
    StreamStatus,
)
from app.services.chat import ChatService

router = APIRouter(prefix="/chats", tags=["Chats"])
logger = structlog.get_logger()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _raise(result: Any) -> Any:
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


# ----------------------------------------------------------------------
# Conversations (declared before ``/{chat_id}`` so paths do not collide)
# ----------------------------------------------------------------------
@router.post(
    "/conversations", response_model=ConversationResponse, status_code=201
)
async def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).create_conversation(current_user.id, payload)
    conversation = _raise(result)
    return conversation_payload(conversation)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    params: dict = Depends(pagination_params),
    status_filter: str | None = Query(None, alias="status"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).get_conversations(
        user_id=current_user.id,
        skip=params["skip"],
        limit=params["limit"],
        search=params.get("search"),
        status=status_filter,
    )
    data = _raise(result)
    return {
        "items": [conversation_payload(item) for item in data["items"]],
        "total": data["total"],
        "page": data["page"],
        "page_size": data["page_size"],
    }


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationResponse
)
async def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).get_conversation(
        current_user.id, conversation_id
    )
    return conversation_payload(_raise(result))


@router.patch(
    "/conversations/{conversation_id}", response_model=ConversationResponse
)
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).update_conversation(
        current_user.id, conversation_id, payload
    )
    return conversation_payload(_raise(result))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).delete_conversation(
        current_user.id, conversation_id
    )
    return _raise(result)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=MessageListResponse,
)
async def list_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).get_messages(
        current_user.id, conversation_id
    )
    data = _raise(result)
    return {
        "items": [message_payload(message) for message in data["items"]],
        "total": data["total"],
    }


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=201,
)
async def send_message(
    conversation_id: str,
    payload: MessageSend,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).send_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        data=payload,
    )
    return message_payload(_raise(result))


@router.post("/conversations/{conversation_id}/cancel")
async def cancel_stream(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    access = await ChatService(session).get_conversation(
        current_user.id, conversation_id
    )
    _raise(access)
    cancelled = ChatService.cancel_stream(conversation_id)
    return {"conversation_id": conversation_id, "cancelled": cancelled}


@router.get(
    "/conversations/{conversation_id}/stream-status",
    response_model=StreamStatus,
)
async def stream_status(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    access = await ChatService(session).get_conversation(
        current_user.id, conversation_id
    )
    _raise(access)
    return ChatService.stream_status(conversation_id)


@router.post("/conversations/{conversation_id}/stream")
async def stream_message(
    conversation_id: str,
    payload: MessageSend,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    """Stream a completion as Server-Sent Events.

    Events: ``start``, ``token``, ``message`` (final persisted message),
    ``done`` and ``error``.  The stream is cancellable via
    ``POST /conversations/{id}/cancel`` or by closing the connection.
    """

    access = await ChatService(session).get_conversation(
        current_user.id, conversation_id
    )
    _raise(access)
    service = ChatService(session)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            async for chunk in service.stream_message(
                user_id=current_user.id,
                conversation_id=conversation_id,
                data=payload,
            ):
                if await request.is_disconnected():
                    ChatService.cancel_stream(conversation_id)
                    break
                if chunk.done:
                    yield {
                        "event": "done",
                        "data": json.dumps(
                            {
                                "finish_reason": chunk.finish_reason,
                                "input_tokens": chunk.input_tokens,
                                "output_tokens": chunk.output_tokens,
                                "model": chunk.model,
                                "provider": chunk.provider,
                            }
                        ),
                    }
                    continue
                yield {
                    "event": "token",
                    "data": json.dumps(
                        {
                            "content": chunk.content,
                            "model": chunk.model,
                            "provider": chunk.provider,
                        }
                    ),
                }

            assistant = await _latest_assistant_message(conversation_id)
            if assistant is not None:
                yield {
                    "event": "message",
                    "data": json.dumps(
                        message_payload(assistant), default=str
                    ),
                }
        except asyncio.CancelledError:
            ChatService.cancel_stream(conversation_id)
            raise
        except ProviderError as exc:
            logger.warning(
                "chat_stream_failed",
                conversation_id=conversation_id,
                code=exc.code,
                error=str(exc),
            )
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "detail": str(exc),
                        "code": exc.code,
                        "status_code": exc.status_code,
                    }
                ),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "chat_stream_unhandled", conversation_id=conversation_id
            )
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "The assistant could not complete the response."}
                ),
            }

    return EventSourceResponse(
        event_generator(), headers=SSE_HEADERS, ping=15, media_type="text/event-stream"
    )


async def _latest_assistant_message(conversation_id: str) -> Message | None:
    """Fetch the persisted assistant message using a fresh session.

    The streaming generator commits its own transaction, so the request
    session must not be reused after the stream completes.
    """

    async with async_session_factory() as session:
        result = await session.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == MessageRole.ASSISTANT,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


# ----------------------------------------------------------------------
# Reusable chat presets
# ----------------------------------------------------------------------
@router.post("", response_model=ChatResponse, status_code=201)
async def create_chat(
    payload: ChatCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).create_chat(current_user.id, payload)
    chat = _raise(result)
    return {
        "id": chat.id,
        "title": chat.title,
        "user_id": chat.user_id,
        "model_id": chat.model_id,
        "provider_name": chat.provider_name,
        "model_name": chat.model_name,
        "system_prompt": chat.system_prompt,
        "temperature": chat.temperature,
        "top_p": chat.top_p,
        "max_tokens": chat.max_tokens,
        "is_active": chat.is_active,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
    }


@router.get("")
async def list_chats(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).get_user_chats(
        current_user.id, skip, limit
    )
    return _raise(result)


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).get_chat(current_user.id, chat_id)
    chat = _raise(result)
    return {
        "id": chat.id,
        "title": chat.title,
        "user_id": chat.user_id,
        "model_id": chat.model_id,
        "provider_name": chat.provider_name,
        "model_name": chat.model_name,
        "system_prompt": chat.system_prompt,
        "temperature": chat.temperature,
        "top_p": chat.top_p,
        "max_tokens": chat.max_tokens,
        "is_active": chat.is_active,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
    }


@router.patch("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    payload: ChatUpdate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).update_chat(
        current_user.id, chat_id, payload
    )
    chat = _raise(result)
    return {
        "id": chat.id,
        "title": chat.title,
        "user_id": chat.user_id,
        "model_id": chat.model_id,
        "provider_name": chat.provider_name,
        "model_name": chat.model_name,
        "system_prompt": chat.system_prompt,
        "temperature": chat.temperature,
        "top_p": chat.top_p,
        "max_tokens": chat.max_tokens,
        "is_active": chat.is_active,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
    }


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await ChatService(session).delete_chat(current_user.id, chat_id)
    return _raise(result)
