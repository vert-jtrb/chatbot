"""Shared transport for OpenAI-compatible provider APIs.

OpenAI, Azure OpenAI, Groq, DeepSeek, LM Studio, vLLM, Ollama's ``/v1``
surface, and the Hugging Face Inference Providers router all expose the same
chat-completions contract.  Implementing the transport once keeps provider
adapters small and guarantees consistent retries, streaming, and error
mapping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from app.providers.ai.base import (
    AIProvider,
    ChatMessage,
    ChatResult,
    ProviderConfig,
    StreamChunk,
)
from app.providers.ai.exceptions import (
    InvalidProviderResponseError,
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(AIProvider):
    provider_type = "openai_compatible"
    display_name = "OpenAI-compatible"
    default_base_url = "https://api.openai.com/v1"
    models_path = "/models"
    chat_path = "/chat/completions"
    embeddings_path = "/embeddings"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=15.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _chat_url(self) -> str:
        return self._url(self.chat_path)

    def _raise_for_response(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            body = response.json()
            message = (
                body.get("error", {}).get("message")
                if isinstance(body.get("error"), dict)
                else body.get("error") or body.get("message") or body.get("detail")
            )
        except (ValueError, AttributeError):
            message = response.text[:500]
        message = str(message or f"HTTP {response.status_code}")
        kwargs = {"provider": self.provider_type}
        if response.status_code in (401, 403):
            raise ProviderAuthenticationError(
                f"Provider rejected the credentials: {message}", **kwargs
            )
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Provider model or endpoint not found: {message}", **kwargs
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(
                f"Provider rate limit exceeded: {message}", **kwargs
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Provider is unavailable: {message}", **kwargs
            )
        raise ProviderError(f"Provider request failed: {message}", **kwargs)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: dict | None = None,
        retries: int | None = None,
    ) -> httpx.Response:
        attempts = max(1, (retries if retries is not None else self.config.max_retries) + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.request(
                    method, url, json=json_body, headers=self._headers()
                )
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt < attempts - 1
                ):
                    delay = _retry_delay(response, attempt)
                    logger.warning(
                        "Retrying provider request",
                        extra={
                            "provider": self.provider_type,
                            "status": response.status_code,
                            "attempt": attempt + 1,
                            "delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                self._raise_for_response(response)
                return response
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                await asyncio.sleep(min(2**attempt, 8))
        if isinstance(last_error, httpx.TimeoutException):
            raise ProviderTimeoutError(
                f"Provider timed out after {self.config.timeout}s",
                provider=self.provider_type,
            ) from last_error
        raise ProviderUnavailableError(
            f"Could not reach provider: {last_error}", provider=self.provider_type
        ) from last_error

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def _messages(
        self, messages: list[ChatMessage], system_prompt: str | None
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        if system_prompt:
            payload.append({"role": "system", "content": system_prompt})
        payload.extend(message.to_openai() for message in messages)
        return payload

    def _payload(
        self,
        messages: list[ChatMessage],
        model: str | None,
        system_prompt: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[dict] | None,
        response_format: dict | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": self._messages(messages, system_prompt),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

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
        response = await self._request(
            "POST",
            self._chat_url(),
            json_body=self._payload(
                messages,
                model,
                system_prompt,
                temperature,
                top_p,
                max_tokens,
                tools,
                response_format,
                stream=False,
            ),
        )
        data = _json(response)
        choices = data.get("choices") or []
        if not choices:
            raise InvalidProviderResponseError(
                "Provider returned no choices", provider=self.provider_type
            )
        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            message=ChatMessage(
                role=message.get("role") or "assistant",
                content=message.get("content") or "",
                tool_calls=message.get("tool_calls"),
            ),
            finish_reason=choice.get("finish_reason"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
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
            response_format,
            stream=True,
        )
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(attempts):
            emitted = False
            try:
                async with self.client.stream(
                    "POST",
                    self._chat_url(),
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if not response.is_success:
                        await response.aread()
                        if (
                            response.status_code in _RETRYABLE_STATUS
                            and attempt < attempts - 1
                        ):
                            await asyncio.sleep(_retry_delay(response, attempt))
                            continue
                        self._raise_for_response(response)
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            yield StreamChunk(done=True, provider=self.provider_type)
                            return
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        usage = event.get("usage") or {}
                        choices = event.get("choices") or []
                        if not choices:
                            if usage:
                                yield StreamChunk(
                                    input_tokens=usage.get("prompt_tokens"),
                                    output_tokens=usage.get("completion_tokens"),
                                    model=event.get("model"),
                                    provider=self.provider_type,
                                )
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        finish = choice.get("finish_reason")
                        if content:
                            emitted = True
                        if content or finish or delta.get("tool_calls"):
                            yield StreamChunk(
                                content=content,
                                done=finish is not None,
                                finish_reason=finish,
                                tool_calls=delta.get("tool_calls"),
                                model=event.get("model"),
                                provider=self.provider_type,
                            )
                    yield StreamChunk(
                        done=True,
                        provider=self.provider_type,
                        model=self.resolve_model(model),
                    )
                    return
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if emitted or attempt >= attempts - 1:
                    if isinstance(exc, httpx.TimeoutException):
                        raise ProviderTimeoutError(
                            "Provider stream timed out", provider=self.provider_type
                        ) from exc
                    raise ProviderUnavailableError(
                        f"Provider stream disconnected: {exc}",
                        provider=self.provider_type,
                    ) from exc
                await asyncio.sleep(min(2**attempt, 8))

    # ------------------------------------------------------------------
    # Models / embeddings
    # ------------------------------------------------------------------
    async def get_models(self) -> list[dict[str, Any]]:
        response = await self._request("GET", self._url(self.models_path))
        data = _json(response)
        entries = data.get("data") if isinstance(data, dict) else data
        if not isinstance(entries, list):
            return []
        models: list[dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, str):
                entry = {"id": entry}
            if not entry.get("id"):
                continue
            models.append(
                {
                    "id": entry["id"],
                    "name": entry.get("name") or entry["id"],
                    "owned_by": entry.get("owned_by"),
                    "created": entry.get("created"),
                    "metadata": entry,
                }
            )
        return models

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        if not texts:
            return []
        response = await self._request(
            "POST",
            self._url(self.embeddings_path),
            json_body={"model": self.resolve_model(model), "input": texts},
        )
        data = _json(response)
        entries = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [entry.get("embedding") for entry in entries]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise InvalidProviderResponseError(
                "Embedding response was incomplete", provider=self.provider_type
            )
        return vectors

    async def health_check(self) -> bool:
        try:
            await self._request("GET", self._url(self.models_path), retries=0)
            return True
        except ProviderError:
            return False


def _json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError as exc:
        raise InvalidProviderResponseError(
            "Provider returned a non-JSON response"
        ) from exc


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(2**attempt, 8)


__all__ = ["OpenAICompatibleProvider"]
