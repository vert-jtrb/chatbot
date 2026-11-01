from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.providers.ai.exceptions import ProviderError
from app.services.provider import ProviderService


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    model: str
    dimensions: int


class LocalHashingEmbeddings:
    """Deterministic offline embeddings for tests and air-gapped development.

    This is explicitly selected with ``EMBEDDING_PROVIDER=local``.  It is not
    a replacement for a semantic embedding model in production, but it gives
    reproducible, normalised vectors without network access.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions or settings.embedding_dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9_]{2,}", text.lower())
        if not tokens:
            return vector
        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend(
            (_stopword_filtered_pair(tokens[index], tokens[index + 1]), 0.5)
            for index in range(len(tokens) - 1)
        )
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "little") % self.dimensions
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class EmbeddingService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.providers = ProviderService(session)

    async def embed(
        self,
        texts: list[str],
        *,
        provider: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult([], provider or "", model or "", dimensions or 0)
        provider_name = (provider or settings.embedding_provider).lower()
        expected_dimensions = dimensions or settings.embedding_dimensions
        if provider_name in {"local", "hashing"}:
            local = LocalHashingEmbeddings(expected_dimensions)
            return EmbeddingResult(
                local.embed(texts),
                "local",
                "local-hashing",
                expected_dimensions,
            )

        if provider_name == "huggingface":
            resolved_model = model or settings.hf_embedding_model
        else:
            resolved_model = model or settings.resolved_embedding_model(provider_name)

        runtime, db_provider, _db_model, resolved_model = await self.providers.resolve(
            provider_name=provider_name, model_name=resolved_model
        )
        try:
            vectors = await runtime.embeddings(texts, model=resolved_model)
        finally:
            await runtime.aclose()
        if not vectors:
            raise ProviderError(
                f"Embedding provider '{provider_name}' returned no vectors"
            )
        actual = len(vectors[0])
        if actual != expected_dimensions:
            raise ProviderError(
                f"Embedding dimension mismatch: model '{resolved_model}' returned "
                f"{actual} dimensions but the knowledge base expects "
                f"{expected_dimensions}. Recreate the knowledge base with a "
                "matching EMBEDDING_DIMENSIONS value.",
                provider=provider_name,
            )
        return EmbeddingResult(
            vectors=vectors,
            provider=db_provider.name if db_provider else provider_name,
            model=resolved_model,
            dimensions=actual,
        )


def _stopword_filtered_pair(left: str, right: str) -> str:
    return f"{left}_{right}"


__all__ = ["EmbeddingResult", "EmbeddingService", "LocalHashingEmbeddings"]
