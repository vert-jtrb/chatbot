"""Conversation orchestration: persistence, RAG, multimodal input and streaming.

The service is provider-agnostic.  It resolves a runtime adapter through
:class:`~app.services.provider.ProviderService`, which in turn is driven by the
database configuration and the provider registry.  Adding a provider never
requires a change here.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.session import async_session_factory
from app.models.audit import AuditLog
from app.models.chat import (
    AttachmentKind,
    Conversation,
    ConversationStatus,
    Message,
    MessageAttachment,
    MessageRole,
)
from app.models.provider import AIModel
from app.models.usage import UsageRecord
from app.providers.ai.base import ChatMessage, StreamChunk
from app.providers.ai.exceptions import ProviderError
from app.repositories.chat import ChatRepository, ConversationRepository, MessageRepository
from app.schemas.chat import ChatCreate, ChatUpdate, ConversationCreate, ConversationUpdate, MessageSend
from app.services.base import ServiceResult
from app.services.documents import DocumentExtractionError, extract_document
from app.services.provider import ProviderService
from app.services.rag import RAGService
from app.services.storage import LocalStorage, StorageError, attachment_kind, decode_data_url

MAX_TITLE_LENGTH = 80


@dataclass(slots=True)
class StreamControl:
    """Cooperative cancellation handle for one in-flight completion."""

    request_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()


class ChatService:
    _streams: dict[str, StreamControl] = {}

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.chat_repo = ChatRepository(session)
        self.conversation_repo = ConversationRepository(session)
        self.message_repo = MessageRepository(session)
        self.storage = LocalStorage()
        self.last_assistant_message: Message | None = None
        self.last_citations: list[dict[str, Any]] = []
        self._persisted = False

    # ------------------------------------------------------------------
    # Chat presets
    # ------------------------------------------------------------------
    async def create_chat(self, user_id: str, data: ChatCreate) -> ServiceResult:
        if data.model_id and not await self._active_model(data.model_id):
            return ServiceResult.error_response("The selected model is not available", 400)
        chat = await self.chat_repo.create(
            user_id=user_id,
            title=data.title.strip(),
            model_id=data.model_id,
            provider_name=data.provider_name,
            model_name=data.model_name,
            system_prompt=data.system_prompt,
            temperature=data.temperature if data.temperature is not None else 0.7,
            top_p=data.top_p if data.top_p is not None else 1.0,
            max_tokens=data.max_tokens,
            metadata_=data.metadata,
        )
        return ServiceResult.ok(chat, status_code=201)

    async def get_user_chats(
        self, user_id: str, skip: int = 0, limit: int = 50
    ) -> ServiceResult:
        chats, total = await self.chat_repo.list_for_user(user_id, skip, limit)
        return ServiceResult.ok(
            {
                "items": chats,
                "total": total,
                "page": skip // limit + 1 if limit else 1,
                "page_size": limit,
            }
        )

    async def get_chat(self, user_id: str, chat_id: str) -> ServiceResult:
        chat = await self.chat_repo.get_for_user(chat_id, user_id)
        if chat is None:
            return ServiceResult.error_response("Chat not found", 404)
        return ServiceResult.ok(chat)

    async def update_chat(self, user_id: str, chat_id: str, data: ChatUpdate) -> ServiceResult:
        chat = await self.chat_repo.get_for_user(chat_id, user_id)
        if chat is None:
            return ServiceResult.error_response("Chat not found", 404)
        payload = data.model_dump(exclude_unset=True)
        if payload.get("model_id") and not await self._active_model(payload["model_id"]):
            return ServiceResult.error_response("The selected model is not available", 400)
        for key, value in payload.items():
            setattr(chat, key, value)
        await self.session.flush()
        return ServiceResult.ok(chat)

    async def delete_chat(self, user_id: str, chat_id: str) -> ServiceResult:
        chat = await self.chat_repo.get_for_user(chat_id, user_id)
        if chat is None:
            return ServiceResult.error_response("Chat not found", 404)
        await self.chat_repo.delete(chat_id, soft=True)
        return ServiceResult.ok({"message": "Chat deleted", "id": chat_id})

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    async def create_conversation(
        self, user_id: str, data: ConversationCreate
    ) -> ServiceResult:
        template = None
        if data.chat_id:
            template = await self.chat_repo.get_for_user(data.chat_id, user_id)
            if template is None:
                return ServiceResult.error_response("Chat not found", 404)
        model_id = data.model_id or (template.model_id if template else None)
        if model_id and not await self._active_model(model_id):
            return ServiceResult.error_response("The selected model is not available", 400)
        knowledge_base_id = data.knowledge_base_id
        if knowledge_base_id:
            owned = await RAGService(self.session).get_knowledge_base(
                user_id, knowledge_base_id
            )
            if not owned.success:
                return owned
        conversation = await self.conversation_repo.create(
            chat_id=data.chat_id,
            user_id=user_id,
            title=(data.title or "New conversation").strip(),
            status=ConversationStatus.ACTIVE,
            provider_name=data.provider_name
            or (template.provider_name if template else None),
            model_name=data.model_name or (template.model_name if template else None),
            model_id=model_id,
            knowledge_base_id=knowledge_base_id,
            rag_enabled=bool(data.rag_enabled and knowledge_base_id),
            retrieval_config=data.retrieval_config,
            system_prompt=(
                data.system_prompt
                if data.system_prompt is not None
                else (template.system_prompt if template else None)
            ),
            temperature=(
                data.temperature
                if data.temperature is not None
                else (template.temperature if template else 0.7)
            ),
            top_p=(
                data.top_p
                if data.top_p is not None
                else (template.top_p if template else 1.0)
            ),
            max_tokens=(
                data.max_tokens
                if data.max_tokens is not None
                else (template.max_tokens if template else None)
            ),
        )
        return ServiceResult.ok(conversation, status_code=201)

    async def get_conversation(self, user_id: str, conversation_id: str) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            return ServiceResult.error_response("Conversation not found", 404)
        return ServiceResult.ok(conversation)

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
        return ServiceResult.ok(
            {
                "items": items,
                "total": total,
                "page": skip // limit + 1 if limit else 1,
                "page_size": limit,
            }
        )

    async def update_conversation(
        self, user_id: str, conversation_id: str, data: ConversationUpdate
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            return ServiceResult.error_response("Conversation not found", 404)
        payload = data.model_dump(exclude_unset=True)
        if payload.get("model_id") and not await self._active_model(payload["model_id"]):
            return ServiceResult.error_response("The selected model is not available", 400)
        if payload.get("knowledge_base_id"):
            owned = await RAGService(self.session).get_knowledge_base(
                user_id, payload["knowledge_base_id"]
            )
            if not owned.success:
                return owned
        for key, value in payload.items():
            setattr(conversation, key, value)
        if not conversation.knowledge_base_id:
            conversation.rag_enabled = False
        await self.session.flush()
        return ServiceResult.ok(conversation)

    async def delete_conversation(
        self, user_id: str, conversation_id: str
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            return ServiceResult.error_response("Conversation not found", 404)
        conversation.status = ConversationStatus.DELETED
        await self.conversation_repo.delete(conversation_id, soft=True)
        self.cancel_stream(conversation_id)
        return ServiceResult.ok(
            {"message": "Conversation deleted", "id": conversation_id}
        )

    async def get_messages(self, user_id: str, conversation_id: str) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            return ServiceResult.error_response("Conversation not found", 404)
        messages, total = await self.message_repo.get_by_conversation(conversation_id)
        return ServiceResult.ok({"items": list(messages), "total": total})

    # ------------------------------------------------------------------
    # Cancellation / stream status
    # ------------------------------------------------------------------
    @classmethod
    def cancel_stream(cls, conversation_id: str) -> bool:
        control = cls._streams.get(conversation_id)
        if control is None:
            return False
        control.cancel()
        return True

    @classmethod
    def stream_status(cls, conversation_id: str) -> dict[str, Any]:
        control = cls._streams.get(conversation_id)
        if control is None:
            return {"conversation_id": conversation_id, "active": False, "request_id": None}
        return {
            "conversation_id": conversation_id,
            "active": True,
            "request_id": control.request_id,
        }

    # ------------------------------------------------------------------
    # Non-streaming completion
    # ------------------------------------------------------------------
    async def send_message(
        self, user_id: str, conversation_id: str, data: MessageSend
    ) -> ServiceResult:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            return ServiceResult.error_response("Conversation not found", 404)
        try:
            prepared = await self._prepare(
                user_id=user_id,
                conversation=conversation,
                data=data,
                persist_user_message=True,
            )
            result = await prepared.runtime.chat(
                messages=prepared.history,
                model=prepared.model_name,
                system_prompt=prepared.system_prompt,
                temperature=conversation.temperature,
                top_p=conversation.top_p,
                max_tokens=conversation.max_tokens,
            )
            assistant = await self._persist_assistant(
                conversation=conversation,
                prepared=prepared,
                content=result.message.content,
                finish_reason=result.finish_reason,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms
                or int((time.perf_counter() - prepared.started) * 1000),
            )
            await self._record_usage(
                user_id=user_id,
                conversation=conversation,
                prepared=prepared,
                input_tokens=assistant.input_tokens or 0,
                output_tokens=assistant.output_tokens or 0,
                latency_ms=assistant.latency_ms or 0,
                status="success",
            )
            await self._audit(
                user_id=user_id,
                action="chat.completion",
                resource="conversation",
                resource_id=conversation_id,
                details={
                    "provider": prepared.provider_name,
                    "model": prepared.model_name,
                    "stream": False,
                },
            )
            self.last_assistant_message = assistant
            self.last_citations = prepared.citations
            return ServiceResult.ok(assistant, status_code=201)
        except ProviderError as exc:
            await self._audit(
                user_id=user_id,
                action="chat.completion",
                resource="conversation",
                resource_id=conversation_id,
                details={"error": str(exc), "code": exc.code},
                success=False,
                severity="error",
            )
            return ServiceResult.error_response(str(exc), exc.status_code)
        except StorageError as exc:
            return ServiceResult.error_response(str(exc), 400)

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------
    async def stream_message(
        self, user_id: str, conversation_id: str, data: MessageSend
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield completion chunks using a dedicated session.

        A request-scoped dependency may already be closing by the time a
        streaming body is iterated, so the generator owns its transaction.
        Partial output is persisted if the client disconnects or cancels.
        """

        async with async_session_factory() as session:
            service = ChatService(session)
            try:
                async for chunk in service._stream(user_id, conversation_id, data):
                    yield chunk
            finally:
                if service._persisted:
                    await session.commit()
                else:
                    await session.rollback()

    async def _stream(
        self, user_id: str, conversation_id: str, data: MessageSend
    ) -> AsyncGenerator[StreamChunk, None]:
        conversation = await self.conversation_repo.get_by_user_id(
            conversation_id, user_id
        )
        if conversation is None:
            raise ProviderError("Conversation not found", provider=None)
        control = StreamControl(request_id=str(uuid4()))
        self._streams[conversation_id] = control
        content = ""
        finish_reason: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        prepared = None
        try:
            prepared = await self._prepare(
                user_id=user_id,
                conversation=conversation,
                data=data,
                persist_user_message=True,
            )
            yield StreamChunk(
                content="",
                done=False,
                model=prepared.model_name,
                provider=prepared.provider_name,
            )
            async for chunk in prepared.runtime.stream_chat(
                messages=prepared.history,
                model=prepared.model_name,
                system_prompt=prepared.system_prompt,
                temperature=conversation.temperature,
                top_p=conversation.top_p,
                max_tokens=conversation.max_tokens,
            ):
                if control.cancelled:
                    finish_reason = "cancelled"
                    break
                if chunk.content:
                    content += chunk.content
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                if chunk.input_tokens is not None:
                    input_tokens = chunk.input_tokens
                if chunk.output_tokens is not None:
                    output_tokens = chunk.output_tokens
                if chunk.content:
                    yield StreamChunk(
                        content=chunk.content,
                        done=False,
                        model=prepared.model_name,
                        provider=prepared.provider_name,
                    )
            if control.cancelled and finish_reason is None:
                finish_reason = "cancelled"
            latency_ms = int((time.perf_counter() - prepared.started) * 1000)
            assistant = await self._persist_assistant(
                conversation=conversation,
                prepared=prepared,
                content=content,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )
            await self._record_usage(
                user_id=user_id,
                conversation=conversation,
                prepared=prepared,
                input_tokens=assistant.input_tokens or 0,
                output_tokens=assistant.output_tokens or 0,
                latency_ms=latency_ms,
                status="cancelled" if finish_reason == "cancelled" else "success",
            )
            await self._audit(
                user_id=user_id,
                action="chat.completion",
                resource="conversation",
                resource_id=conversation_id,
                details={
                    "provider": prepared.provider_name,
                    "model": prepared.model_name,
                    "stream": True,
                    "cancelled": finish_reason == "cancelled",
                },
            )
            self.last_assistant_message = assistant
            self.last_citations = prepared.citations
            yield StreamChunk(
                content="",
                done=True,
                finish_reason=finish_reason or "stop",
                input_tokens=assistant.input_tokens,
                output_tokens=assistant.output_tokens,
                model=prepared.model_name,
                provider=prepared.provider_name,
            )
        except ProviderError as exc:
            if prepared is not None:
                await self._record_usage(
                    user_id=user_id,
                    conversation=conversation,
                    prepared=prepared,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                    latency_ms=int((time.perf_counter() - prepared.started) * 1000),
                    status="error",
                    error_code=exc.code,
                    error_message=str(exc),
                )
            await self._audit(
                user_id=user_id,
                action="chat.completion",
                resource="conversation",
                resource_id=conversation_id,
                details={"error": str(exc), "code": exc.code},
                success=False,
                severity="error",
            )
            raise
        finally:
            if prepared is not None and content and not self.last_assistant_message:
                # Client disconnected or the provider raised mid-stream: keep
                # whatever was generated instead of losing the partial answer.
                assistant = await self._persist_assistant(
                    conversation=conversation,
                    prepared=prepared,
                    content=content,
                    finish_reason=finish_reason or "interrupted",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=int((time.perf_counter() - prepared.started) * 1000),
                )
                self.last_assistant_message = assistant
            self._streams.pop(conversation_id, None)
            self._persisted = self.last_assistant_message is not None

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------
    async def _prepare(
        self,
        *,
        user_id: str,
        conversation: Conversation,
        data: MessageSend,
        persist_user_message: bool,
    ) -> "_PreparedGeneration":
        started = time.perf_counter()
        provider_name = data.provider_name or conversation.provider_name
        model_row_id = data.model_id or conversation.model_id
        model_name_hint = data.model_name or conversation.model_name
        providers = ProviderService(self.session)
        try:
            runtime, db_provider, db_model, resolved_model = await providers.resolve(
                provider_name=provider_name if not model_row_id else None,
                model_id=model_row_id,
                model_name=model_name_hint,
            )
        except ProviderError:
            raise
        if not resolved_model:
            raise ProviderError(
                "No model was selected for this conversation",
                provider=provider_name,
            )

        user_content = (data.content or "").strip()
        image_parts: list[str] = []
        audio_payload: str | None = None
        attachment_text: list[str] = []
        stored_attachments: list[tuple[str, str, str, int, str, str]] = []
        for attachment in data.attachments:
            stored = self.storage.save_data_url(
                kind="attachments",
                filename=attachment.file_name,
                data_url=attachment.data_url,
                allowed=(
                    settings.allowed_upload_extensions
                    + settings.allowed_image_extensions
                    + settings.allowed_audio_extensions
                ),
            )
            kind = attachment_kind(stored.mime_type, stored.file_name)
            stored_attachments.append(
                (
                    kind,
                    stored.file_name,
                    stored.storage_path,
                    stored.file_size,
                    stored.mime_type,
                    stored.checksum,
                )
            )
            if kind == AttachmentKind.IMAGE.value:
                image_parts.append(attachment.data_url)
                continue
            if kind == AttachmentKind.AUDIO.value:
                _mime, raw = decode_data_url(attachment.data_url)
                audio_payload = base64.b64encode(raw).decode("ascii")
                continue
            try:
                extracted = await asyncio.to_thread(
                    extract_document, Path(stored.storage_path), stored.file_name
                )
                if extracted.text.strip():
                    attachment_text.append(
                        f"[Attachment: {stored.file_name}]\n{extracted.text[:12000]}"
                    )
            except DocumentExtractionError:
                attachment_text.append(
                    f"[Attachment: {stored.file_name}] (binary file, content unavailable)"
                )

        if attachment_text:
            user_content = "\n\n".join(
                part for part in [user_content, *attachment_text] if part
            )
        if not user_content and not image_parts and not audio_payload:
            raise ProviderError("A message must contain text or an attachment")
        if not user_content and image_parts:
            user_content = "Describe this image."

        citations: list[dict[str, Any]] = []
        system_prompt = conversation.system_prompt
        if conversation.rag_enabled and conversation.knowledge_base_id:
            context, citations, failure = await RAGService(self.session).build_context(
                user_id,
                kb_id=conversation.knowledge_base_id,
                query=user_content,
                config=conversation.retrieval_config,
            )
            if failure is not None:
                raise ProviderError(
                    failure.error or "Knowledge-base retrieval failed",
                    provider="rag",
                )
            if context:
                system_prompt = (
                    f"{system_prompt}\n\n{context}"
                    if system_prompt
                    else context
                )

        history = await self._history(
            conversation.id,
            current_content=user_content,
            images=image_parts or None,
            audio=audio_payload,
        )
        user_message = None
        if persist_user_message:
            user_message = await self.message_repo.create(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=user_content,
                parent_id=data.parent_id,
                provider_name=db_provider.name if db_provider else provider_name,
                model_name=resolved_model,
                model_id=model_row_id,
                provider_id=db_provider.id if db_provider else None,
            )
            for (
                kind,
                file_name,
                storage_path,
                file_size,
                mime_type,
                checksum,
            ) in stored_attachments:
                self.session.add(
                    MessageAttachment(
                        message_id=user_message.id,
                        user_id=user_id,
                        kind=kind,
                        file_name=file_name,
                        storage_path=storage_path,
                        file_size=file_size,
                        mime_type=mime_type,
                        storage_type=settings.storage_type.value,
                        metadata_={"sha256": checksum},
                    )
                )
            await self.session.flush()
            await self._touch_conversation(conversation, user_content)

        if db_provider is not None:
            conversation.provider_name = db_provider.name
        conversation.model_id = model_row_id or conversation.model_id
        conversation.model_name = resolved_model
        await self.session.flush()

        return _PreparedGeneration(
            runtime=runtime,
            provider_name=db_provider.name if db_provider else (provider_name or "unknown"),
            model_name=resolved_model,
            model_row=db_model,
            system_prompt=system_prompt,
            history=history,
            citations=citations,
            user_message=user_message,
            started=started,
        )

    async def _history(
        self,
        conversation_id: str,
        *,
        current_content: str,
        images: list[str] | None,
        audio: str | None,
    ) -> list[ChatMessage]:
        messages = await self.message_repo.recent_for_context(
            conversation_id, limit=settings.chat_context_messages
        )
        history: list[ChatMessage] = []
        for index, message in enumerate(messages):
            is_last = index == len(messages) - 1
            history.append(
                ChatMessage(
                    role=str(message.role),
                    content=(
                        current_content
                        if is_last and message.role == MessageRole.USER
                        else message.content
                    ),
                    images=images if is_last and message.role == MessageRole.USER else None,
                    audio=audio if is_last and message.role == MessageRole.USER else None,
                    tool_calls=message.tool_calls,
                    tool_call_id=message.tool_call_id,
                )
            )
        if not history:
            history.append(
                ChatMessage(role="user", content=current_content, images=images, audio=audio)
            )
        return history

    async def _touch_conversation(
        self, conversation: Conversation, content: str
    ) -> None:
        conversation.last_message_at = datetime.now(UTC)
        conversation.message_count = conversation.message_count + 1
        if conversation.title in {"", "New chat", "New conversation"}:
            conversation.title = _title_from(content)
        await self.session.flush()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    async def _persist_assistant(
        self,
        *,
        conversation: Conversation,
        prepared: "_PreparedGeneration",
        content: str,
        finish_reason: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        latency_ms: int | None,
    ) -> Message:
        content = content.strip()
        if not content:
            content = "[The provider returned an empty response]"
        input_tokens = input_tokens or _estimate_tokens(
            "".join(message.content for message in prepared.history)
        )
        output_tokens = output_tokens or _estimate_tokens(content)
        assistant = await self.message_repo.create(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=content,
            model_name=prepared.model_name,
            provider_name=prepared.provider_name,
            model_id=prepared.model_row.id if prepared.model_row else None,
            citations=prepared.citations or None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason or "stop",
        )
        conversation.last_message_at = datetime.now(UTC)
        conversation.message_count = conversation.message_count + 1
        conversation.token_count = conversation.token_count + input_tokens + output_tokens
        await self.session.flush()
        self.last_assistant_message = assistant
        return assistant

    async def _record_usage(
        self,
        *,
        user_id: str,
        conversation: Conversation,
        prepared: "_PreparedGeneration",
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        cost = 0.0
        if prepared.model_row is not None:
            if prepared.model_row.input_price_per_million:
                cost += (
                    input_tokens / 1_000_000
                ) * prepared.model_row.input_price_per_million
            if prepared.model_row.output_price_per_million:
                cost += (
                    output_tokens / 1_000_000
                ) * prepared.model_row.output_price_per_million
        self.session.add(
            UsageRecord(
                user_id=user_id,
                conversation_id=conversation.id,
                provider=prepared.provider_name,
                model=prepared.model_name,
                operation="chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=round(cost, 8),
                latency_ms=latency_ms,
                status=status,
                error_code=error_code,
                error_message=error_message,
            )
        )
        await self.session.flush()

    async def _audit(
        self,
        *,
        user_id: str,
        action: str,
        resource: str,
        resource_id: str | None,
        details: dict[str, Any] | None = None,
        success: bool = True,
        severity: str = "info",
    ) -> None:
        self.session.add(
            AuditLog(
                user_id=user_id,
                action=action,
                resource=resource,
                resource_id=resource_id,
                details=details,
                severity=severity,
                success=success,
            )
        )
        await self.session.flush()

    async def _active_model(self, model_row_id: str) -> bool:
        result = await self.session.execute(
            select(AIModel.id).where(
                AIModel.id == model_row_id, AIModel.is_active.is_(True)
            )
        )
        return result.scalar_one_or_none() is not None

    async def load_attachments(
        self, user_id: str, attachment_ids: Sequence[str]
    ) -> list[MessageAttachment]:
        if not attachment_ids:
            return []
        result = await self.session.execute(
            select(MessageAttachment).where(
                MessageAttachment.id.in_(list(attachment_ids)),
                MessageAttachment.user_id == user_id,
            )
        )
        return list(result.scalars().all())


@dataclass(slots=True)
class _PreparedGeneration:
    runtime: Any
    provider_name: str
    model_name: str
    model_row: AIModel | None
    system_prompt: str | None
    history: list[ChatMessage]
    citations: list[dict[str, Any]]
    user_message: Message | None
    started: float


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _title_from(content: str) -> str:
    cleaned = " ".join(content.split())
    if not cleaned:
        return "New conversation"
    return cleaned[:MAX_TITLE_LENGTH]


__all__ = ["ChatService", "StreamControl"]
