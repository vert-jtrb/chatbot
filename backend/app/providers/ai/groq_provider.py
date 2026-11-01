from __future__ import annotations

from app.core.config import settings
from app.providers.ai.base import ModelCapabilities, ProviderConfig
from app.providers.ai.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    provider_type = "groq"
    display_name = "Groq"
    default_base_url = "https://api.groq.com/openai/v1"
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
            name="groq",
            provider_type=self.provider_type,
            api_key=settings.groq_api_key,
            default_model=settings.groq_default_model,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
        )
        config.base_url = config.base_url or self.default_base_url
        config.api_key = config.api_key or settings.groq_api_key
        config.default_model = config.default_model or settings.groq_default_model
        super().__init__(config)


__all__ = ["GroqProvider"]
