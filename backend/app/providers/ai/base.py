from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from app.providers.ai.exceptions import UnsupportedCapabilityError


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str = ""
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    images: list[str] | None = None
    audio: str | None = None

    def to_openai(self) -> dict[str, Any]:
        if not self.images and not self.audio:
            payload: dict[str, Any] = {"role": self.role, "content": self.content}
        else:
            parts: list[dict[str, Any]] = []
            if self.content:
                parts.append({"type": "text", "text": self.content})
            for image in self.images or []:
                parts.append({"type": "image_url", "image_url": {"url": image}})
            if self.audio:
                parts.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": self.audio, "format": "wav"},
                    }
                )
            payload = {"role": self.role, "content": parts}
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(slots=True)
class StreamChunk:
    content: str = ""
    done: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    provider: str | None = None
    tool_calls: list[dict] | None = None


@dataclass(slots=True)
class ChatResult:
    message: ChatMessage
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    provider: str | None = None
    latency_ms: int | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class ModelCapabilities:
    streaming: bool = True
    tools: bool = False
    vision: bool = False
    embeddings: bool = False
    audio: bool = False
    json_mode: bool = False
    max_context_length: int = 4096


@dataclass(slots=True)
class ProviderConfig:
    name: str
    provider_type: str
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    default_inference_provider: str | None = None
    timeout: float = 120.0
    max_retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class AIProvider(ABC):
    """Base contract every provider adapter implements.

    Adding a provider only requires a new adapter plus a registry entry; the
    chat/RAG services never branch on concrete provider names.
    """

    provider_type: str = ""
    display_name: str = ""
    capabilities: ModelCapabilities = ModelCapabilities()
    supports_model_discovery: bool = True
    requires_api_key: bool = True

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        raise NotImplementedError

    async def get_models(self) -> list[dict[str, Any]]:
        return []

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        raise UnsupportedCapabilityError(
            f"{self.provider_type} does not support embeddings",
            provider=self.provider_type,
        )

    async def transcribe(self, audio: bytes, model: str | None = None) -> str:
        raise UnsupportedCapabilityError(
            f"{self.provider_type} does not support speech recognition",
            provider=self.provider_type,
        )

    async def synthesize(self, text: str, model: str | None = None, voice: str | None = None) -> bytes:
        raise UnsupportedCapabilityError(
            f"{self.provider_type} does not support speech synthesis",
            provider=self.provider_type,
        )

    async def health_check(self) -> bool:
        try:
            await self.get_models()
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        return None

    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def resolve_model(self, model: str | None) -> str:
        resolved = model or self.config.default_model
        if not resolved:
            raise ValueError(f"No model configured for provider {self.provider_type}")
        return resolved


__all__ = [
    "AIProvider",
    "ChatMessage",
    "ChatResult",
    "ModelCapabilities",
    "ProviderConfig",
    "StreamChunk",
]
