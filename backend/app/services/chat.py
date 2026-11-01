from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import Message, MessageRole
from app.providers.ai.base import StreamChunk
from app.repositories.chat import (
    ChatRepository,
    ConversationRepository,
    MessageRepository,
)
from app.schemas.chat import ConversationCreate, MessageSend
from app.services.ai import AIService
from app.services.base import ServiceResult


class ChatService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.chat_repo = ChatRepository(session)
        self.conversation_repo = ConversationRepository(session)
        self.message_repo = MessageRepository(session)
        self.last_assistant_message: Message | None = None

    async def create_chat(
        self, user_id: str, title: str, **kwargs: object
    ) -> ServiceResult:
        chat = await self.chat_repo.create(
            user_id=user_id, title=title, **kwargs
        )
        return ServiceResult.ok(chat, status_code=201)

    async def get_user_chats(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> ServiceResult:
        chats, total = await self.chat_repo.get_many(
            skip=skip, limit=limit, filters={"user_id": user_id}
        )
        return ServiceResult.ok({"items": chats, "total": total})

    async def create_conversation(
        self, user_id: str, data: ConversationCreate
    ) -> ServiceResult:
        chat = await self.chat_repo.get(data.chat_id)
        if not chat or chat.user_id != user_id:
            return ServiceResult.error_response("Chat not found", 404)

        conversation = await self.conversation_repo.create(
            chat_id=data.chat_id,
            user_id=user_id,
            title=data.title,
            model_id=(
                data.model_id if data.model_id is not None else chat.model_id
            ),
            system_prompt=(
                data.system_prompt
                if data.system_prompt is not None
                else chat.system_prompt
            ),
            temperature=(
                data.temperature
                if data.temperature is not None
                else chat.temperature
            ),
            top_p=data.top_p if data.top_p is not None else chat.top_p,
            max_tokens=(
                data.max_tokens if data.max_tokens is not None else chat.max_tokens
            ),
        )
        return ServiceResult.ok(conversation, status_code=201)

    async def get_conversation(
        self, user_id: str, conversation_id: str
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if not conversation:
            return ServiceResult.error_response("Conversation not found", 404)
        return ServiceResult.ok(conversation)

    async def get_messages(
        self, user_id: str, conversation_id: str
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if not conversation:
            return ServiceResult.error_response("Conversation not found", 404)
        messages, _ = await self.message_repo.get_by_conversation(
            conversation_id
        )
        return ServiceResult.ok(messages)

    async def get_conversations(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
        status: str | None = None,
    ) -> ServiceResult:
        items, total = await self.conversation_repo.get_by_user(
            user_id, skip, limit, search, status
        )
        return ServiceResult.ok({
            "items": items,
            "total": total,
            "page": skip // limit + 1,
            "page_size": limit,
        })

    async def send_message(
        self,
        user_id: str,
        conversation_id: str,
        data: MessageSend,
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if not conversation:
            return ServiceResult.error_response("Conversation not found", 404)

        message = await self.message_repo.create(
            conversation_id=conversation_id,
            role=MessageRole(data.role.value),
            content=data.content,
            parent_id=data.parent_id,
        )

        await self.conversation_repo.update(
            conversation_id,
            last_message_at=datetime.now(UTC),
        )
        await self.conversation_repo.update_message_count(conversation_id)

        return ServiceResult.ok(message, status_code=201)

    async def stream_message(
        self,
        user_id: str,
        conversation_id: str,
        data: MessageSend,
    ) -> AsyncGenerator[StreamChunk]:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if not conversation:
            raise ValueError("Conversation not found")

        self.last_assistant_message = None
        await self.message_repo.create(
            conversation_id=conversation_id,
            role=MessageRole(data.role.value),
            content=data.content,
            parent_id=data.parent_id,
        )
        await self.conversation_repo.update(
            conversation_id,
            last_message_at=datetime.now(UTC),
        )
        await self.conversation_repo.update_message_count(conversation_id)

        messages, _ = await self.message_repo.get_by_conversation(
            conversation_id
        )

        ai_service = AIService(self.session)
        full_content = ""
        finish_reason = None

        try:
            async for chunk in ai_service.stream_chat(
                messages=list(messages),
                model_id=conversation.model_id,
                system_prompt=conversation.system_prompt,
                temperature=conversation.temperature,
                top_p=conversation.top_p,
                max_tokens=conversation.max_tokens,
            ):
                full_content += chunk.content
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                yield chunk
        finally:
            if full_content:
                self.last_assistant_message = await self.message_repo.create(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_content,
                    model_id=conversation.model_id,
                    finish_reason=finish_reason,
                )
            await self.conversation_repo.update(
                conversation_id,
                last_message_at=datetime.now(UTC),
            )
            await self.conversation_repo.update_message_count(conversation_id)
