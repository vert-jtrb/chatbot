from __future__ import annotations

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    provider_type = "openai"
    display_name = "OpenAI"
    default_base_url = settings.openai_api_base
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        json_mode=True,
        max_context_length=128000,
    )

    def __init__(self, config=None):
        if config is None:
            from app.providers.ai.base import ProviderConfig

            config = ProviderConfig(
                name="openai",
                provider_type=self.provider_type,
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
                default_model=settings.openai_default_model,
                timeout=settings.provider_timeout,
                max_retries=settings.provider_max_retries,
            )
        config.base_url = config.base_url or settings.openai_api_base
        config.api_key = config.api_key or settings.openai_api_key
        config.default_model = config.default_model or settings.openai_default_model
        super().__init__(config)


__all__ = ["OpenAIProvider"]
