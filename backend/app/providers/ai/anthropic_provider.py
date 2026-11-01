from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.core.config import settings
from app.providers.ai.base import (
    AIProvider,
    ChatMessage,
    ChatResult,
    ModelCapabilities,
    ProviderConfig,
    StreamChunk,
)
from app.providers.ai.exceptions import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class AnthropicProvider(AIProvider):
    provider_type = "anthropic"
    display_name = "Anthropic"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=False,
        json_mode=False,
        max_context_length=200000,
    )
    default_base_url = "https://api.anthropic.com/v1"
    api_version = "2023-06-01"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="anthropic",
            provider_type=self.provider_type,
            api_key=settings.anthropic_api_key,
            default_model=settings.anthropic_default_model,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
        )
        config.base_url = (config.base_url or self.default_base_url).rstrip("/")
        config.api_key = config.api_key or settings.anthropic_api_key
        config.default_model = config.default_model or settings.anthropic_default_model
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=15.0)
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": self.api_version,
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        return headers

    def _raise(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body = response.json()
            message = body.get("error", {}).get("message") or body.get("error")
        except ValueError:
            message = response.text[:500]
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError(
                f"Anthropic rejected the API key: {message}", provider=self.provider_type
            )
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Anthropic model not found: {message}", provider=self.provider_type
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(
                f"Anthropic rate limit exceeded: {message}",
                provider=self.provider_type,
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Anthropic is unavailable: {message}", provider=self.provider_type
            )
        raise ProviderError(
            f"Anthropic request failed: {message}", provider=self.provider_type
        )

    def _content(self, message: ChatMessage) -> str | list[dict[str, Any]]:
        if not message.images:
            return message.content
        blocks: list[dict[str, Any]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for image in message.images:
            if image.startswith("data:"):
                header, data = image.split(",", 1)
                media_type = header.split(";")[0].removeprefix("data:")
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": image}}
                )
        return blocks

    def _messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id or "",
                                "content": message.content,
                            }
                        ],
                    }
                )
                continue
            entry: dict[str, Any] = {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": self._content(message),
            }
            if message.tool_calls:
                blocks = (
                    entry["content"]
                    if isinstance(entry["content"], list)
                    else [{"type": "text", "text": entry["content"]}]
                )
                for call in message.tool_calls:
                    function = call.get("function") or {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", ""),
                            "name": function.get("name", ""),
                            "input": _json_loads(function.get("arguments") or "{}"),
                        }
                    )
                entry["content"] = blocks
            converted.append(entry)
        return converted

    def _payload(
        self,
        messages: list[ChatMessage],
        model: str | None,
        system_prompt: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[dict] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": self._messages(messages),
            "max_tokens": max_tokens or 4096,
        }
        system = system_prompt or next(
            (message.content for message in messages if message.role == "system"), None
        )
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if tools:
            payload["tools"] = [
                {
                    "name": tool.get("function", {}).get("name", ""),
                    "description": tool.get("function", {}).get("description", ""),
                    "input_schema": tool.get("function", {}).get(
                        "parameters", {"type": "object"}
                    ),
                }
                for tool in tools
            ]
        if stream:
            payload["stream"] = True
        return payload

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = max(1, self.config.max_retries + 1)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.post(
                    f"{self.config.base_url}/messages",
                    json=payload,
                    headers=self._headers(),
                )
                if response.status_code in _RETRYABLE and attempt < attempts - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                self._raise(response)
                return response.json()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt >= attempts - 1:
                    break
                await asyncio.sleep(min(2**attempt, 8))
        if isinstance(last, httpx.TimeoutException):
            raise ProviderTimeoutError(
                "Anthropic timed out", provider=self.provider_type
            ) from last
        raise ProviderUnavailableError(
            f"Could not reach Anthropic: {last}", provider=self.provider_type
        ) from last

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        data = await self._post(
            self._payload(
                messages,
                model,
                system_prompt,
                temperature,
                top_p,
                max_tokens,
                tools,
                stream=False,
            )
        )
        blocks = data.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if block.get("type") == "text"
        )
        tool_calls = [
            {
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            }
            for block in blocks
            if block.get("type") == "tool_use"
        ]
        usage = data.get("usage") or {}
        return ChatResult(
            message=ChatMessage(
                role="assistant", content=text, tool_calls=tool_calls or None
            ),
            finish_reason=data.get("stop_reason"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            model=data.get("model") or self.resolve_model(model),
            provider=self.provider_type,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        payload = self._payload(
            messages,
            model,
            system_prompt,
            temperature,
            top_p,
            max_tokens,
            tools,
            stream=True,
        )
        async with self.client.stream(
            "POST",
            f"{self.config.base_url}/messages",
            json=payload,
            headers=self._headers(),
        ) as response:
            if not response.is_success:
                await response.aread()
                self._raise(response)
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "message_start":
                    usage = (event.get("message") or {}).get("usage") or {}
                    yield StreamChunk(
                        input_tokens=usage.get("input_tokens"),
                        provider=self.provider_type,
                    )
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    text = delta.get("text") or delta.get("partial_json") or ""
                    if text:
                        yield StreamChunk(
                            content=text, provider=self.provider_type
                        )
                elif event_type == "message_delta":
                    usage = event.get("usage") or {}
                    yield StreamChunk(
                        done=True,
                        finish_reason=(event.get("delta") or {}).get("stop_reason"),
                        output_tokens=usage.get("output_tokens"),
                        provider=self.provider_type,
                    )
                elif event_type == "message_stop":
                    return

    async def get_models(self) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.config.base_url}/models", headers=self._headers()
        )
        self._raise(response)
        data = response.json()
        return [
            {
                "id": model.get("id"),
                "name": model.get("display_name") or model.get("id"),
                "owned_by": "anthropic",
                "created": model.get("created_at"),
                "metadata": model,
            }
            for model in data.get("data") or []
            if model.get("id")
        ]

    async def health_check(self) -> bool:
        try:
            await self.get_models()
            return True
        except ProviderError:
            return False


def _json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {}


__all__ = ["AnthropicProvider"]
