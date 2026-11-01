from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings
from app.providers.ai.huggingface.capabilities import (
    best_provider,
    detect_capabilities,
    pricing,
    provider_names,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HuggingFaceModel:
    id: str
    name: str
    owned_by: str | None = None
    pipeline_tag: str | None = None
    context_length: int = 4096
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    supports_json_mode: bool = False
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    providers: list[str] = field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    private: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owned_by": self.owned_by,
            "pipeline_tag": self.pipeline_tag,
            "context_length": self.context_length,
            "input_modalities": self.input_modalities,
            "output_modalities": self.output_modalities,
            "supports_tools": self.supports_tools,
            "supports_vision": self.supports_vision,
            "supports_embeddings": self.supports_embeddings,
            "supports_audio": self.supports_audio,
            "supports_json_mode": self.supports_json_mode,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "providers": self.providers,
            "downloads": self.downloads,
            "likes": self.likes,
            "private": self.private,
            "metadata": self.metadata,
        }


class HuggingFaceCatalog:
    """Discovers models from the router and the Hugging Face Hub.

    The router catalogue carries live provider pricing/capability metadata,
    while the Hub catalogue exposes the long tail of task-specific models.
    Results are cached briefly and deduplicated by model id.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        router_url: str | None = None,
        hub_url: str | None = None,
        ttl: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.token = token or settings.hf_token
        self.router_url = (router_url or settings.hf_router_url).rstrip("/")
        self.hub_url = (hub_url or settings.hf_hub_url).rstrip("/")
        self.ttl = ttl if ttl is not None else settings.hf_model_cache_ttl
        self.timeout = timeout or settings.hf_timeout
        self._cache: dict[str, tuple[float, list[HuggingFaceModel]]] = {}
        self._lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def list_router_models(self) -> list[HuggingFaceModel]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.router_url}/models", headers=self._headers()
            )
        if response.status_code == 401:
            raise PermissionError("Hugging Face token is invalid")
        response.raise_for_status()
        entries = response.json().get("data") or []
        return [self._from_router(entry) for entry in entries if entry.get("id")]

    async def list_hub_models(
        self,
        *,
        pipeline_tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
        inference_provider: str = "all",
    ) -> list[HuggingFaceModel]:
        params: dict[str, Any] = {
            "limit": min(max(limit, 1), 500),
            "sort": "downloads",
            "direction": -1,
            "expand[]": [
                "inferenceProviderMapping",
                "pipeline_tag",
                "downloads",
                "likes",
                "library_name",
                "tags",
            ],
        }
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag
        if search:
            params["search"] = search
        if inference_provider:
            params["inference_provider"] = inference_provider
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                f"{self.hub_url}/api/models", params=params, headers=self._headers()
            )
        response.raise_for_status()
        entries = response.json()
        return [self._from_hub(entry) for entry in entries if entry.get("id")]

    async def all_models(self, *, refresh: bool = False) -> list[HuggingFaceModel]:
        key = "all"
        cached = self._cache.get(key)
        if not refresh and cached and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        async with self._lock:
            cached = self._cache.get(key)
            if not refresh and cached and time.monotonic() - cached[0] < self.ttl:
                return cached[1]
            router_models: list[HuggingFaceModel] = []
            try:
                router_models = await self.list_router_models()
            except (httpx.HTTPError, PermissionError) as exc:
                logger.warning("Could not load Hugging Face router models: %s", exc)
            hub_models: list[HuggingFaceModel] = []
            try:
                hub_models = await self.list_hub_models(limit=200)
            except httpx.HTTPError as exc:
                logger.warning("Could not load Hugging Face Hub models: %s", exc)
            merged = {model.id: model for model in hub_models}
            for model in router_models:
                existing = merged.get(model.id)
                if existing:
                    # Router metadata is authoritative for live inference.
                    existing.providers = model.providers
                    existing.supports_tools = model.supports_tools
                    existing.supports_vision = model.supports_vision
                    existing.supports_json_mode = model.supports_json_mode
                    existing.context_length = model.context_length
                    existing.input_price_per_million = model.input_price_per_million
                    existing.output_price_per_million = model.output_price_per_million
                    existing.metadata.update(model.metadata)
                else:
                    merged[model.id] = model
            models = sorted(
                merged.values(),
                key=lambda item: (item.downloads or 0, item.likes or 0),
                reverse=True,
            )
            self._cache[key] = (time.monotonic(), models)
            return models

    async def search(
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
        models = await self.all_models(refresh=refresh)
        query_lower = (query or "").lower().strip()
        results: list[HuggingFaceModel] = []
        for model in models:
            if query_lower and query_lower not in model.id.lower():
                continue
            if pipeline_tag and model.pipeline_tag != pipeline_tag:
                continue
            if supports_vision is True and not model.supports_vision:
                continue
            if supports_tools is True and not model.supports_tools:
                continue
            if supports_embeddings is True and not model.supports_embeddings:
                continue
            if provider and provider not in model.providers:
                continue
            results.append(model)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _from_router(entry: dict[str, Any]) -> HuggingFaceModel:
        caps = detect_capabilities(entry)
        provider = best_provider(entry) or {}
        input_price, output_price = pricing(entry)
        return HuggingFaceModel(
            id=entry["id"],
            name=entry.get("name") or entry["id"],
            owned_by=entry.get("owned_by"),
            pipeline_tag=entry.get("pipeline_tag") or "text-generation",
            context_length=caps.max_context_length,
            input_modalities=(entry.get("architecture") or {}).get(
                "input_modalities", ["text"]
            ),
            output_modalities=(entry.get("architecture") or {}).get(
                "output_modalities", ["text"]
            ),
            supports_tools=caps.tools,
            supports_vision=caps.vision,
            supports_embeddings=caps.embeddings,
            supports_audio=caps.audio,
            supports_json_mode=caps.json_mode,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
            providers=provider_names(entry),
            metadata={
                "router": entry,
                "selected_provider": provider.get("provider"),
                "first_token_latency_ms": provider.get("first_token_latency_ms"),
                "throughput": provider.get("throughput"),
            },
        )

    @staticmethod
    def _from_hub(entry: dict[str, Any]) -> HuggingFaceModel:
        caps = detect_capabilities(entry)
        mapping = entry.get("inferenceProviderMapping") or {}
        providers = list(mapping.keys()) if isinstance(mapping, dict) else []
        return HuggingFaceModel(
            id=entry["id"],
            name=entry["id"].split("/")[-1],
            owned_by=(entry["id"].split("/")[0] if "/" in entry["id"] else None),
            pipeline_tag=entry.get("pipeline_tag"),
            context_length=caps.max_context_length,
            input_modalities=(entry.get("architecture") or {}).get(
                "input_modalities", []
            ),
            output_modalities=(entry.get("architecture") or {}).get(
                "output_modalities", ["text"]
            ),
            supports_tools=caps.tools,
            supports_vision=caps.vision,
            supports_embeddings=caps.embeddings,
            supports_audio=caps.audio,
            supports_json_mode=caps.json_mode,
            providers=providers,
            downloads=entry.get("downloads"),
            likes=entry.get("likes"),
            private=bool(entry.get("private")),
            metadata={"hub": entry},
        )


__all__ = ["HuggingFaceCatalog", "HuggingFaceModel"]
