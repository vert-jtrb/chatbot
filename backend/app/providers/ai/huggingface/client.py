"""Low-level Hugging Face client (router, Hub, embeddings, speech)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.config import settings
from app.providers.ai.base import (
    ChatMessage,
    ChatResult,
    ProviderConfig,
    StreamChunk,
)
from app.providers.ai.exceptions import ProviderAuthenticationError
from app.providers.ai.huggingface.embeddings import HuggingFaceEmbeddings
from app.providers.ai.huggingface.models import HuggingFaceCatalog, HuggingFaceModel
from app.providers.ai.huggingface.multimodal import (
    synthesize as _synthesize,
)
from app.providers.ai.huggingface.multimodal import (
    transcribe as _transcribe,
)
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class HuggingFaceClient:
    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str | None = None,
        endpoint_url: str | None = None,
        default_provider: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        hub_url: str | None = None,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.token = token or settings.hf_token
        self.endpoint_url = (endpoint_url or settings.hf_endpoint_url or "").rstrip("/")
        self.router_url = (base_url or settings.hf_router_url).rstrip("/")
        self.hub_url = (hub_url or settings.hf_hub_url).rstrip("/")
        self.default_provider = default_provider or settings.hf_default_provider
        self.timeout = timeout or settings.hf_timeout
        self.max_retries = (
            max_retries if max_retries is not None else settings.hf_max_retries
        )
        self.embedding_model = embedding_model or settings.hf_embedding_model
        self.embedding_provider = (
            embedding_provider or settings.hf_embedding_provider
        )
        self.embedding_dimensions = (
            embedding_dimensions or settings.embedding_dimensions
        )
        self._catalog = HuggingFaceCatalog(
            token=self.token,
            router_url=self.router_url,
            hub_url=self.hub_url,
            timeout=self.timeout,
        )
        self._transport: OpenAICompatibleProvider | None = None
        self._embeddings: HuggingFaceEmbeddings | None = None

    @property
    def chat_base_url(self) -> str:
        if self.endpoint_url:
            return (
                self.endpoint_url
                if self.endpoint_url.endswith("/v1")
                else f"{self.endpoint_url}/v1"
            )
        return self.router_url

    @property
    def transport(self) -> OpenAICompatibleProvider:
        if self._transport is None:
            if not self.token:
                raise ProviderAuthenticationError(
                    "HF_TOKEN is required for Hugging Face chat inference. "
                    "Create a read token at https://huggingface.co/settings/tokens",
                    provider="huggingface",
                )
            self._transport = OpenAICompatibleProvider(
                ProviderConfig(
                    name="huggingface",
                    provider_type="huggingface",
                    api_key=self.token,
                    base_url=self.chat_base_url,
                    default_model=settings.hf_default_model,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            )
        return self._transport

    @property
    def embeddings_client(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                token=self.token,
                model=self.embedding_model,
                provider=self.embedding_provider,
                dimensions=self.embedding_dimensions,
                timeout=self.timeout,
                endpoint_url=self.endpoint_url or None,
            )
        return self._embeddings

    @property
    def catalog(self) -> HuggingFaceCatalog:
        return self._catalog

    def resolve_model(self, model: str | None) -> str:
        resolved = model or settings.hf_default_model
        if not resolved:
            raise ValueError("No Hugging Face model configured")
        if self.endpoint_url or ":" in resolved:
            return resolved
        provider = self.default_provider
        if provider and provider not in {"auto", "none"}:
            return f"{resolved}:{provider}"
        return resolved

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
        result = await self.transport.chat(
            messages,
            model=self.resolve_model(model),
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        )
        result.provider = "huggingface"
        return result

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
        resolved = self.resolve_model(model)
        async for chunk in self.transport.stream_chat(
            messages,
            model=resolved,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            response_format=response_format,
        ):
            chunk.provider = "huggingface"
            chunk.model = resolved
            yield chunk

    async def list_models(self, *, refresh: bool = False) -> list[HuggingFaceModel]:
        return await self.catalog.all_models(refresh=refresh)

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
    ) -> list[HuggingFaceModel]:
        return await self.catalog.search(
            query,
            pipeline_tag=pipeline_tag,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
            supports_embeddings=supports_embeddings,
            provider=provider,
            limit=limit,
            refresh=refresh,
        )

    async def model_details(self, model_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.hub_url}/api/models/{model_id}",
                headers=self._headers(),
            )
        response.raise_for_status()
        return response.json()

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        client = self.embeddings_client
        if model:
            client.model = model
        return await client.embed(texts)

    async def transcribe(self, audio: bytes, model: str | None = None) -> str:
        return await asyncio.to_thread(
            _transcribe,
            audio,
            token=self.token,
            model=model,
            endpoint_url=self.endpoint_url or None,
        )

    async def synthesize(
        self, text: str, model: str | None = None, voice: str | None = None
    ) -> bytes:
        return await asyncio.to_thread(
            _synthesize,
            text,
            token=self.token,
            model=model,
            endpoint_url=self.endpoint_url or None,
        )

    async def validate_token(self) -> dict[str, Any]:
        if not self.token:
            raise ProviderAuthenticationError(
                "HF_TOKEN is not configured", provider="huggingface"
            )
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{self.hub_url}/api/whoami-v2", headers=self._headers()
            )
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError(
                "Hugging Face token is invalid", provider="huggingface"
            )
        response.raise_for_status()
        data = response.json()
        return {
            "name": data.get("name"),
            "type": data.get("type"),
            "email": data.get("email"),
            "plan": (data.get("plan") or "free") if isinstance(data, dict) else "free",
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.router_url}/models", headers=self._headers()
                )
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        if self._transport is not None:
            await self._transport.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}


__all__ = ["HuggingFaceClient"]
