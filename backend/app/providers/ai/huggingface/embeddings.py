"""Hugging Face embedding support.

Embeddings use ``huggingface_hub.InferenceClient`` so feature-extraction
models can be routed through Inference Providers or a dedicated endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.providers.ai.exceptions import (
    InvalidProviderResponseError,
    ProviderAuthenticationError,
    ProviderUnavailableError,
)


class HuggingFaceEmbeddings:
    def __init__(
        self,
        token: str | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
        dimensions: int | None = None,
        timeout: float | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        self.token = token or settings.hf_token
        self.model = model or settings.hf_embedding_model
        self.provider = provider or settings.hf_embedding_provider
        self.dimensions = dimensions or settings.embedding_dimensions
        self.timeout = timeout or settings.hf_timeout
        self.endpoint_url = endpoint_url
        self._client: InferenceClient | None = None

    @property
    def client(self) -> InferenceClient:
        if self._client is None:
            if not self.token:
                raise ProviderAuthenticationError(
                    "HF_TOKEN is required for Hugging Face embeddings",
                    provider="huggingface",
                )
            kwargs: dict[str, Any] = {
                "token": self.token,
                "timeout": self.timeout,
            }
            if self.endpoint_url:
                kwargs["base_url"] = self.endpoint_url
            elif self.provider and self.provider != "auto":
                kwargs["provider"] = self.provider
            self._client = InferenceClient(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            result = self.client.feature_extraction(
                texts, model=self.model
            )
        except Exception as exc:  # huggingface_hub raises many HTTP errors
            message = str(exc)
            if "401" in message or "403" in message:
                raise ProviderAuthenticationError(
                    "Hugging Face rejected the token for embeddings",
                    provider="huggingface",
                ) from exc
            raise ProviderUnavailableError(
                f"Hugging Face embedding request failed: {message}",
                provider="huggingface",
            ) from exc
        vectors = _to_vectors(result)
        if not vectors:
            raise InvalidProviderResponseError(
                "Hugging Face returned no embeddings", provider="huggingface"
            )
        if self.dimensions and vectors and len(vectors[0]) != self.dimensions:
            raise InvalidProviderResponseError(
                "Embedding model returned "
                f"{len(vectors[0])} dimensions but EMBEDDING_DIMENSIONS is "
                f"{self.dimensions}. Set EMBEDDING_DIMENSIONS to match the model.",
                provider="huggingface",
            )
        return vectors


def _to_vectors(result: Any) -> list[list[float]]:
    if result is None:
        return []
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, dict):
        result = result.get("embeddings") or result.get("data") or []
    if isinstance(result, list) and result and isinstance(result[0], (int, float)):
        return [[float(value) for value in result]]
    vectors: list[list[float]] = []
    for row in result:
        if isinstance(row, dict):
            row = row.get("embedding", row)
        if hasattr(row, "tolist"):
            row = row.tolist()
        vectors.append([float(value) for value in row])
    return vectors


__all__ = ["HuggingFaceEmbeddings"]
