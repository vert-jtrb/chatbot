from __future__ import annotations

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities, ProviderConfig
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_type = "deepseek"
    display_name = "DeepSeek"
    default_base_url = "https://api.deepseek.com/v1"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=False,
        embeddings=False,
        json_mode=True,
        max_context_length=131072,
    )

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="deepseek",
            provider_type=self.provider_type,
            api_key=settings.deepseek_api_key,
            default_model=settings.deepseek_default_model,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
        )
        config.base_url = config.base_url or self.default_base_url
        config.api_key = config.api_key or settings.deepseek_api_key
        config.default_model = config.default_model or settings.deepseek_default_model
        super().__init__(config)


__all__ = ["DeepSeekProvider"]
