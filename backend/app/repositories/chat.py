from __future__ import annotations

from collections.abc import Sequence
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.database.base import utc_now
from app.models.chat import Chat, Conversation, Message
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository[Chat]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Chat)

    async def get_with_conversations(self, chat_id: str) -> Chat | None:
        stmt = (
            select(Chat)
            .options(joinedload(Chat.conversations))
            .where(Chat.id == chat_id, Chat.is_deleted.is_(False))
        )
        result = await self.session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def get_for_user(self, chat_id: str, user_id: str) -> Chat | None:
        stmt = select(Chat).where(
            Chat.id == chat_id,
            Chat.user_id == user_id,
            Chat.is_deleted.is_(False),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_user(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> tuple[Sequence[Chat], int]:
        where = (Chat.user_id == user_id, Chat.is_deleted.is_(False))
        total = await self.session.scalar(
            select(func.count()).select_from(Chat).where(*where)
        )
        result = await self.session.execute(
            select(Chat)
            .where(*where)
            .order_by(Chat.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)


class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Conversation)

    async def get_by_user_id(
        self, conversation_id: str, user_id: str
    ) -> Conversation | None:
        stmt = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.is_deleted.is_(False),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
        search: str | None = None,
        status: str | None = None,
    ) -> tuple[Sequence[Conversation], int]:
        where = [Conversation.user_id == user_id, Conversation.is_deleted.is_(False)]
        if search:
            where.append(Conversation.title.ilike(f"%{search}%"))
        if status:
            where.append(Conversation.status == status)
        total = await self.session.scalar(
            select(func.count()).select_from(Conversation).where(*where)
        )
        stmt = (
            select(Conversation)
            .where(*where)
            .order_by(
                Conversation.is_pinned.desc(),
                Conversation.last_message_at.desc().nullslast(),
                Conversation.updated_at.desc(),
            )
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all(), int(total or 0)

    async def touch(
        self, conversation_id: str, *, token_count: int, title_hint: str | None = None
    ) -> None:
        conversation = await self.get(conversation_id)
        if conversation is None:
            return
        conversation.last_message_at = utc_now()
        conversation.message_count = conversation.message_count + 2
        conversation.token_count = conversation.token_count + token_count
        if title_hint and conversation.title in {"", "New chat", "New conversation"}:
            conversation.title = title_hint[:80]
        await self.session.flush()


class MessageRepository(BaseRepository[Message]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Message)

    async def get_by_conversation(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 200,
        ascending: bool = True,
    ) -> tuple[Sequence[Message], int]:
        total = await self.session.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.conversation_id == conversation_id)
        )
        order = Message.created_at.asc() if ascending else Message.created_at.desc()
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(order)
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all(), int(total or 0)

    async def recent_for_context(
        self, conversation_id: str, limit: int = 30
    ) -> list[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    async def get_owned(self, message_id: str, user_id: str) -> Message | None:
        stmt = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.id == message_id,
                Conversation.user_id == user_id,
                Conversation.is_deleted.is_(False),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


__all__ = ["ChatRepository", "ConversationRepository", "MessageRepository"]
