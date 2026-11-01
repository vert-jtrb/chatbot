from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities, ProviderConfig
from app.providers.ai.exceptions import ProviderError
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class AzureOpenAIProvider(OpenAICompatibleProvider):
    provider_type = "azure_openai"
    display_name = "Azure OpenAI"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        json_mode=True,
        max_context_length=128000,
    )

    def __init__(self, config: ProviderConfig | None = None) -> None:
        extra = (config.extra if config else None) or {}
        endpoint = extra.get("endpoint") or settings.azure_openai_endpoint
        config = config or ProviderConfig(
            name="azure_openai",
            provider_type=self.provider_type,
            api_key=settings.azure_openai_api_key,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
            extra=extra,
        )
        self.endpoint = (config.base_url or endpoint or "").rstrip("/")
        if not self.endpoint:
            raise ProviderError(
                "Azure OpenAI endpoint is not configured", provider=self.provider_type
            )
        self.api_version = (
            config.extra.get("api_version")
            or extra.get("api_version")
            or settings.azure_openai_api_version
        )
        self.deployment = (
            config.extra.get("deployment")
            or extra.get("deployment")
            or settings.azure_openai_default_deployment
            or config.default_model
        )
        config.base_url = self.endpoint
        config.api_key = config.api_key or settings.azure_openai_api_key
        config.default_model = config.default_model or self.deployment
        self._active_model: str | None = None
        super().__init__(config)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["api-key"] = self.config.api_key
        return headers

    def _deployment(self, model: str | None) -> str:
        return model or self._active_model or self.deployment or self.config.default_model or ""

    def _chat_url(self) -> str:
        deployment = self._deployment(None)
        if not deployment:
            raise ProviderError(
                "Azure OpenAI deployment is not configured",
                provider=self.provider_type,
            )
        return (
            f"{self.endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _url(self, path: str) -> str:
        if path == self.embeddings_path:
            deployment = self._deployment(None)
            return (
                f"{self.endpoint}/openai/deployments/{deployment}"
                f"/embeddings?api-version={self.api_version}"
            )
        if path == self.models_path:
            return f"{self.endpoint}/openai/deployments?api-version={self.api_version}"
        return f"{self.endpoint}{path}"

    def _payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = super()._payload(*args, **kwargs)
        self._active_model = payload.pop("model", None)
        return payload

    async def get_models(self) -> list[dict[str, Any]]:
        if not self.deployment and not self.config.default_model:
            return []
        name = self.deployment or self.config.default_model
        return [
            {
                "id": name,
                "name": name,
                "owned_by": "azure-openai",
                "metadata": {"deployment": name},
            }
        ]


__all__ = ["AzureOpenAIProvider"]
