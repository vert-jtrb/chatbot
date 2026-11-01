import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies.auth import get_current_user
from app.api.dependencies.pagination import pagination_params
from app.database.session import get_session
from app.models.user import User
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
    MessageSend,
)
from app.services.chat import ChatService

router = APIRouter(prefix="/chats", tags=["Chats"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse, status_code=201)
async def create_chat(
    request: ChatCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = ChatService(session)
    result = await service.create_chat(
        user_id=current_user.id,
        title=request.title,
        model_id=request.model_id,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    return result.data


@router.get("")
async def list_chats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
) -> dict:
    service = ChatService(session)
    result = await service.get_user_chats(current_user.id, skip, limit)
    return result.data


@router.post("/conversations", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    request: ConversationCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = ChatService(session)
    result = await service.create_conversation(current_user.id, request)
    if not result.success:
        raise HTTPException(
            status_code=result.status_code, detail=result.error
        )
    return result.data


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    params: dict = Depends(pagination_params),
) -> dict:
    service = ChatService(session)
    result = await service.get_conversations(
        user_id=current_user.id,
        skip=params["skip"],
        limit=params["limit"],
        search=params.get("search"),
    )
    return result.data


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    service = ChatService(session)
    result = await service.get_messages(current_user.id, conversation_id)
    if not result.success:
        raise HTTPException(
            status_code=result.status_code, detail=result.error
        )
    return result.data


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
    status_code=201,
)
async def send_message(
    conversation_id: str,
    request: MessageSend,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = ChatService(session)
    result = await service.send_message(
        user_id=current_user.id,
        conversation_id=conversation_id,
        data=request,
    )
    if not result.success:
        raise HTTPException(
            status_code=result.status_code, detail=result.error
        )
    return result.data


@router.post("/conversations/{conversation_id}/stream")
async def stream_message(
    conversation_id: str,
    request: MessageSend,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    service = ChatService(session)
    access = await service.get_conversation(current_user.id, conversation_id)
    if not access.success:
        raise HTTPException(
            status_code=access.status_code, detail=access.error
        )

    async def event_generator() -> AsyncGenerator[dict]:
        try:
            async for chunk in service.stream_message(
                user_id=current_user.id,
                conversation_id=conversation_id,
                data=request,
            ):
                yield {
                    "event": "token",
                    "data": json.dumps(asdict(chunk)),
                }

            assistant_message = service.last_assistant_message
            if assistant_message:
                yield {
                    "event": "message",
                    "data": json.dumps(
                        MessageResponse.model_validate(
                            assistant_message
                        ).model_dump(mode="json")
                    ),
                }
            yield {"event": "done", "data": ""}
        except Exception:
            logger.exception(
                "Chat response stream failed for conversation %s",
                conversation_id,
            )
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "The assistant could not complete the response."}
                ),
            }

    return EventSourceResponse(event_generator())
