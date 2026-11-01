from __future__ import annotations

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities, ProviderConfig
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    provider_type = "vllm"
    display_name = "vLLM"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        json_mode=True,
        max_context_length=131072,
    )
    requires_api_key = False

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="vllm",
            provider_type=self.provider_type,
            api_key=settings.vllm_api_key,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
        )
        base = config.base_url or settings.vllm_base_url
        config.base_url = base if base.endswith("/v1") else f"{base.rstrip('/')}/v1"
        super().__init__(config)


__all__ = ["VLLMProvider"]
