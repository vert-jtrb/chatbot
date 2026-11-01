"""Streaming helpers shared by the Hugging Face adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from app.providers.ai.base import StreamChunk


async def with_provider_metadata(
    stream: AsyncGenerator[StreamChunk, None],
    *,
    model: str,
    provider: str,
) -> AsyncGenerator[StreamChunk, None]:
    """Annotate chunks so persisted messages retain routing information."""

    async for chunk in stream:
        if chunk.model is None:
            chunk.model = model
        if chunk.provider is None:
            chunk.provider = provider
        yield chunk


__all__ = ["with_provider_metadata"]
