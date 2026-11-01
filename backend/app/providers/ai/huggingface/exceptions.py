"""Map Hugging Face Hub/HTTP failures onto the shared provider errors."""

from __future__ import annotations

from typing import Any

from app.providers.ai.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)


def translate_hf_error(exc: Exception) -> ProviderError:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    message = str(exc)
    if status in (401, 403) or "Invalid credentials" in message:
        return ProviderAuthenticationError(
            "Hugging Face rejected the token; verify HF_TOKEN", provider="huggingface"
        )
    if status == 404 or "not found" in message.lower():
        return ModelNotFoundError(
            f"Hugging Face model not found: {message}", provider="huggingface"
        )
    if status == 429:
        return ProviderRateLimitError(
            "Hugging Face rate limit exceeded; retry later", provider="huggingface"
        )
    return ProviderUnavailableError(
        f"Hugging Face request failed: {message}", provider="huggingface"
    )


def error_payload(exc: Exception) -> dict[str, Any]:
    return translate_hf_error(exc).to_dict()


__all__ = ["error_payload", "translate_hf_error"]
