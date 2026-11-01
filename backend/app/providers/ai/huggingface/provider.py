from __future__ import annotations

from collections.abc import AsyncGenerator

from app.core.config import settings
from app.providers.ai.base import (
    AIProvider,
    ChatMessage,
    ChatResult,
    ModelCapabilities,
    ProviderConfig,
    StreamChunk,
)
from app.providers.ai.huggingface.client import HuggingFaceClient


class HuggingFaceProvider(AIProvider):
    """First-class Hugging Face Inference Providers adapter.

    Model ids may include a routing suffix (``model:provider``,
    ``model:fastest``, ``model:cheapest``).  When no suffix is present the
    configured default inference provider is applied automatically.
    """

    provider_type = "huggingface"
    display_name = "Hugging Face"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        audio=True,
        json_mode=True,
        max_context_length=131072,
    )
    requires_api_key = True

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        extra = config.extra or {}
        self.client = HuggingFaceClient(
            token=config.api_key or settings.hf_token,
            base_url=config.base_url or settings.hf_router_url,
            endpoint_url=extra.get("endpoint_url") or settings.hf_endpoint_url,
            default_provider=config.default_inference_provider,
            timeout=config.timeout,
            max_retries=config.max_retries,
            embedding_model=extra.get("embedding_model"),
            embedding_provider=extra.get("embedding_provider"),
            embedding_dimensions=extra.get("embedding_dimensions"),
        )

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
        return await self.client.chat(
            messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )

    async def stream_chat(
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
        async for chunk in self.client.stream_chat(
            messages,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        ):
            yield chunk

    async def get_models(self) -> list[dict]:
        models = await self.client.list_models()
        return [model.to_dict() for model in models]

    async def search_models(
        self,
        query: str | None = None,
        *,
        pipeline_tag: str | None = None,
        supports_vision: bool | None = None,
        supports_tools: bool | None = None,
        supports_embeddings: bool | None = None,
        provider: str | None = None,
        limit: int = 50,
        refresh: bool = False,
    ) -> list[dict]:
        models = await self.client.search_models(
            query,
            pipeline_tag=pipeline_tag,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
            supports_embeddings=supports_embeddings,
            provider=provider,
            limit=limit,
            refresh=refresh,
        )
        return [model.to_dict() for model in models]

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        return await self.client.embeddings(texts, model=model)

    async def transcribe(self, audio: bytes, model: str | None = None) -> str:
        return await self.client.transcribe(audio, model=model)

    async def synthesize(
        self, text: str, model: str | None = None, voice: str | None = None
    ) -> bytes:
        return await self.client.synthesize(text, model=model, voice=voice)

    async def health_check(self) -> bool:
        return await self.client.health_check()

    async def validate_token(self) -> dict:
        return await self.client.validate_token()

    async def aclose(self) -> None:
        await self.client.aclose()


__all__ = ["HuggingFaceProvider"]
