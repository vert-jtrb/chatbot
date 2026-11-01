from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.models.provider import ProviderType
from app.providers.ai.anthropic_provider import AnthropicProvider
from app.providers.ai.azure_openai_provider import AzureOpenAIProvider
from app.providers.ai.base import AIProvider, ProviderConfig
from app.providers.ai.deepseek_provider import DeepSeekProvider
from app.providers.ai.gemini_provider import GeminiProvider
from app.providers.ai.groq_provider import GroqProvider
from app.providers.ai.huggingface.provider import HuggingFaceProvider
from app.providers.ai.lm_studio_provider import LMStudioProvider
from app.providers.ai.ollama_provider import OllamaProvider
from app.providers.ai.openai_compatible import OpenAICompatibleProvider
from app.providers.ai.openai_provider import OpenAIProvider
from app.providers.ai.vllm_provider import VLLMProvider

PROVIDER_CLASSES: dict[str, type[AIProvider]] = {
    ProviderType.OPENAI.value: OpenAIProvider,
    ProviderType.AZURE_OPENAI.value: AzureOpenAIProvider,
    ProviderType.ANTHROPIC.value: AnthropicProvider,
    ProviderType.GEMINI.value: GeminiProvider,
    ProviderType.GROQ.value: GroqProvider,
    ProviderType.DEEPSEEK.value: DeepSeekProvider,
    ProviderType.HUGGINGFACE.value: HuggingFaceProvider,
    ProviderType.OLLAMA.value: OllamaProvider,
    ProviderType.LM_STUDIO.value: LMStudioProvider,
    ProviderType.VLLM.value: VLLMProvider,
    ProviderType.CUSTOM.value: OpenAICompatibleProvider,
}

_LOCAL_TYPES = {
    ProviderType.OLLAMA.value,
    ProviderType.LM_STUDIO.value,
    ProviderType.VLLM.value,
}


def default_config(provider_type: str) -> ProviderConfig:
    key = provider_type.lower()
    return ProviderConfig(
        name=key,
        provider_type=key,
        api_key=_env_key(key),
        base_url=_env_base_url(key),
        default_model=_env_model(key),
        timeout=settings.provider_timeout,
        max_retries=settings.provider_max_retries,
        extra=_env_extra(key),
    )


def create_provider(config: ProviderConfig) -> AIProvider:
    provider_class = PROVIDER_CLASSES.get(config.provider_type.lower())
    if provider_class is None:
        provider_class = OpenAICompatibleProvider
    return provider_class(config)


def _env_key(provider_type: str) -> str | None:
    return {
        ProviderType.OPENAI.value: settings.openai_api_key,
        ProviderType.AZURE_OPENAI.value: settings.azure_openai_api_key,
        ProviderType.ANTHROPIC.value: settings.anthropic_api_key,
        ProviderType.GEMINI.value: settings.google_api_key,
        ProviderType.GROQ.value: settings.groq_api_key,
        ProviderType.DEEPSEEK.value: settings.deepseek_api_key,
        ProviderType.HUGGINGFACE.value: settings.hf_token,
        ProviderType.VLLM.value: settings.vllm_api_key,
        ProviderType.LM_STUDIO.value: settings.lm_studio_api_key,
    }.get(provider_type)


def _env_base_url(provider_type: str) -> str | None:
    return {
        ProviderType.OPENAI.value: settings.openai_api_base,
        ProviderType.AZURE_OPENAI.value: settings.azure_openai_endpoint,
        ProviderType.OLLAMA.value: settings.ollama_base_url,
        ProviderType.VLLM.value: settings.vllm_base_url,
        ProviderType.LM_STUDIO.value: settings.lm_studio_base_url,
        ProviderType.HUGGINGFACE.value: settings.hf_router_url,
    }.get(provider_type)


def _env_model(provider_type: str) -> str | None:
    return {
        ProviderType.OPENAI.value: settings.openai_default_model,
        ProviderType.AZURE_OPENAI.value: settings.azure_openai_default_deployment,
        ProviderType.ANTHROPIC.value: settings.anthropic_default_model,
        ProviderType.GEMINI.value: settings.google_default_model,
        ProviderType.GROQ.value: settings.groq_default_model,
        ProviderType.DEEPSEEK.value: settings.deepseek_default_model,
        ProviderType.HUGGINGFACE.value: settings.hf_default_model,
        ProviderType.OLLAMA.value: settings.ollama_default_model,
    }.get(provider_type)


def _env_extra(provider_type: str) -> dict[str, Any]:
    if provider_type == ProviderType.AZURE_OPENAI.value:
        return {
            "endpoint": settings.azure_openai_endpoint,
            "api_version": settings.azure_openai_api_version,
            "deployment": settings.azure_openai_default_deployment,
        }
    if provider_type == ProviderType.HUGGINGFACE.value:
        return {
            "endpoint_url": settings.hf_endpoint_url,
            "embedding_model": settings.hf_embedding_model,
            "embedding_provider": settings.hf_embedding_provider,
        }
    return {}


def has_credentials(provider_type: str) -> bool:
    if provider_type in _LOCAL_TYPES:
        return True
    if provider_type == ProviderType.HUGGINGFACE.value:
        return bool(settings.hf_token)
    extra = _env_extra(provider_type)
    if provider_type == ProviderType.AZURE_OPENAI.value:
        return bool(_env_key(provider_type) and extra.get("endpoint"))
    return bool(_env_key(provider_type))


class ProviderRegistry:
    """Process-level registry of environment-configured providers.

    Database-managed provider configurations are resolved by
    :class:`app.services.provider.ProviderService`; this registry powers
    discovery endpoints and backwards-compatible lookups.
    """

    _providers: dict[str, AIProvider] = {}

    @classmethod
    def register(cls, provider: AIProvider) -> None:
        cls._providers[provider.config.name] = provider

    @classmethod
    def get(cls, name: str) -> AIProvider:
        provider = cls._providers.get(name)
        if not provider:
            raise ValueError(f"Unknown provider: {name}")
        return provider

    @classmethod
    def get_all(cls) -> dict[str, AIProvider]:
        return cls._providers

    @classmethod
    def available_types(cls) -> list[dict[str, Any]]:
        return [
            {
                "type": provider_type,
                "display_name": provider_class.display_name or provider_type,
                "requires_api_key": provider_class.requires_api_key,
                "capabilities": provider_class.capabilities,
                "configured": has_credentials(provider_type),
            }
            for provider_type, provider_class in PROVIDER_CLASSES.items()
        ]

    @classmethod
    def initialize(cls) -> None:
        cls._providers.clear()
        for provider_type in PROVIDER_CLASSES:
            if not has_credentials(provider_type):
                continue
            try:
                cls.register(create_provider(default_config(provider_type)))
            except Exception:
                # A misconfigured optional provider must not prevent startup.
                continue


__all__ = [
    "PROVIDER_CLASSES",
    "ProviderRegistry",
    "create_provider",
    "default_config",
    "has_credentials",
]
