"""Provider errors mapped to stable API error codes."""

from __future__ import annotations


class ProviderError(RuntimeError):
    code = "provider_error"
    status_code = 502

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider

    def to_dict(self) -> dict[str, str]:
        payload = {"code": self.code, "message": str(self)}
        if self.provider:
            payload["provider"] = self.provider
        return payload


class ProviderAuthenticationError(ProviderError):
    code = "provider_authentication_failed"
    status_code = 401


class ProviderRateLimitError(ProviderError):
    code = "provider_rate_limited"
    status_code = 429


class ProviderTimeoutError(ProviderError):
    code = "provider_timeout"
    status_code = 504


class ProviderUnavailableError(ProviderError):
    code = "provider_unavailable"
    status_code = 503


class ModelNotFoundError(ProviderError):
    code = "model_not_found"
    status_code = 404


class UnsupportedCapabilityError(ProviderError):
    code = "unsupported_capability"
    status_code = 400


class InvalidProviderResponseError(ProviderError):
    code = "invalid_provider_response"
    status_code = 502


__all__ = [
    "InvalidProviderResponseError",
    "ModelNotFoundError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "UnsupportedCapabilityError",
]
