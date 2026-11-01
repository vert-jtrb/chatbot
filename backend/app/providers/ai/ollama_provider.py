from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities, ProviderConfig
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    """Ollama exposes both a native API and an OpenAI-compatible surface."""

    provider_type = "ollama"
    display_name = "Ollama"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        json_mode=True,
        max_context_length=32768,
    )
    requires_api_key = False

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="ollama",
            provider_type=self.provider_type,
            api_key=None,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
            extra={"native_base_url": settings.ollama_base_url},
        )
        native = (config.extra or {}).get("native_base_url") or settings.ollama_base_url
        self.native_base_url = native.rstrip("/").removesuffix("/v1")
        base = config.base_url or f"{self.native_base_url}/v1"
        if not base.rstrip("/").endswith("/v1"):
            base = f"{base.rstrip('/')}/v1"
        config.base_url = base
        config.default_model = config.default_model or settings.ollama_default_model
        super().__init__(config)

    async def get_models(self) -> list[dict[str, Any]]:
        response = await self._request("GET", f"{self.native_base_url}/api/tags", retries=0)
        data = response.json()
        models: list[dict[str, Any]] = []
        for item in data.get("models") or []:
            name = item.get("name") or item.get("model")
            if not name:
                continue
            details = item.get("details") or {}
            models.append(
                {
                    "id": name,
                    "name": name,
                    "owned_by": "ollama",
                    "context_length": details.get("context_length"),
                    "metadata": item,
                }
            )
        return models

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        response = await self._request(
            "POST",
            f"{self.native_base_url}/api/embed",
            json_body={"model": self.resolve_model(model), "input": texts},
        )
        data = response.json()
        return [list(map(float, vector)) for vector in data.get("embeddings") or []]

    async def health_check(self) -> bool:
        try:
            response = await self._request(
                "GET", f"{self.native_base_url}/api/tags", retries=0
            )
            return response.status_code < 500
        except Exception:
            return False


__all__ = ["OllamaProvider"]
