from __future__ import annotations

import asyncio
import json
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

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class GeminiProvider(AIProvider):
    provider_type = "gemini"
    display_name = "Google Gemini"
    capabilities = ModelCapabilities(
        streaming=True,
        tools=True,
        vision=True,
        embeddings=True,
        json_mode=True,
        max_context_length=1048576,
    )
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: ProviderConfig | None = None) -> None:
        config = config or ProviderConfig(
            name="gemini",
            provider_type=self.provider_type,
            api_key=settings.google_api_key,
            default_model=settings.google_default_model,
            timeout=settings.provider_timeout,
            max_retries=settings.provider_max_retries,
        )
        config.base_url = (config.base_url or self.default_base_url).rstrip("/")
        config.api_key = config.api_key or settings.google_api_key
        config.default_model = config.default_model or settings.google_default_model
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

    def _model(self, model: str | None) -> str:
        return self.resolve_model(model).removeprefix("models/")

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
                f"Gemini rejected the API key: {message}", provider=self.provider_type
            )
        if response.status_code == 404:
            raise ModelNotFoundError(
                f"Gemini model not found: {message}", provider=self.provider_type
            )
        if response.status_code == 429:
            raise ProviderRateLimitError(
                f"Gemini rate limit exceeded: {message}", provider=self.provider_type
            )
        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Gemini is unavailable: {message}", provider=self.provider_type
            )
        raise ProviderError(
            f"Gemini request failed: {message}", provider=self.provider_type
        )

    def _parts(self, message: ChatMessage) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        for image in message.images or []:
            if image.startswith("data:"):
                header, data = image.split(",", 1)
                mime = header.split(";")[0].removeprefix("data:")
                parts.append({"inlineData": {"mimeType": mime, "data": data}})
            else:
                parts.append({"fileData": {"fileUri": image}})
        return parts

    def _contents(self, messages: list[ChatMessage]) -> tuple[list[dict], dict | None]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": self._parts(message)})
        system = (
            {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
        )
        return contents, system

    def _payload(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
        tools: list[dict] | None,
    ) -> dict[str, Any]:
        contents, system = self._contents(messages)
        if system_prompt:
            system = {"parts": [{"text": system_prompt}]}
        generation: dict[str, Any] = {}
        if temperature is not None:
            generation["temperature"] = temperature
        if top_p is not None:
            generation["topP"] = top_p
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = system
        if generation:
            payload["generationConfig"] = generation
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.get("function", {}).get("name", ""),
                            "description": tool.get("function", {}).get(
                                "description", ""
                            ),
                            "parameters": tool.get("function", {}).get(
                                "parameters", {"type": "object", "properties": {}}
                            ),
                        }
                        for tool in tools
                    ]
                }
            ]
        return payload

    async def _post(
        self, model: str, payload: dict[str, Any], *, method: str = "generateContent"
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}/models/{model}:{method}"
        attempts = max(1, self.config.max_retries + 1)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    params={"key": self.config.api_key or ""},
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
                "Gemini timed out", provider=self.provider_type
            ) from last
        raise ProviderUnavailableError(
            f"Could not reach Gemini: {last}", provider=self.provider_type
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
        resolved = self._model(model)
        data = await self._post(
            resolved,
            self._payload(
                messages, system_prompt, temperature, top_p, max_tokens, tools
            ),
        )
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(
                "Gemini returned no candidates", provider=self.provider_type
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        function_calls = [
            {
                "id": part.get("functionCall", {}).get("name"),
                "type": "function",
                "function": {
                    "name": part.get("functionCall", {}).get("name"),
                    "arguments": json.dumps(part.get("functionCall", {}).get("args", {})),
                },
            }
            for part in parts
            if part.get("functionCall")
        ]
        usage = data.get("usageMetadata") or {}
        return ChatResult(
            message=ChatMessage(
                role="assistant", content=text, tool_calls=function_calls or None
            ),
            finish_reason=candidate.get("finishReason"),
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            model=resolved,
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
        resolved = self._model(model)
        payload = self._payload(
            messages, system_prompt, temperature, top_p, max_tokens, tools
        )
        url = f"{self.config.base_url}/models/{resolved}:streamGenerateContent"
        async with self.client.stream(
            "POST",
            url,
            json=payload,
            params={"alt": "sse", "key": self.config.api_key or ""},
        ) as response:
            if not response.is_success:
                await response.aread()
                self._raise(response)
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
                usage = event.get("usageMetadata") or {}
                for candidate in event.get("candidates") or []:
                    parts = (candidate.get("content") or {}).get("parts") or []
                    text = "".join(part.get("text", "") for part in parts)
                    finish = candidate.get("finishReason")
                    if text or finish or usage:
                        yield StreamChunk(
                            content=text,
                            done=finish is not None,
                            finish_reason=finish,
                            input_tokens=usage.get("promptTokenCount"),
                            output_tokens=usage.get("candidatesTokenCount"),
                            model=resolved,
                            provider=self.provider_type,
                        )

    async def get_models(self) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"{self.config.base_url}/models",
            params={"key": self.config.api_key or ""},
        )
        self._raise(response)
        data = response.json()
        return [
            {
                "id": model.get("name", "").removeprefix("models/"),
                "name": model.get("displayName") or model.get("name", ""),
                "owned_by": "google",
                "context_length": model.get("inputTokenLimit"),
                "metadata": model,
            }
            for model in data.get("models") or []
            if "generateContent" in (model.get("supportedGenerationMethods") or [])
        ]

    async def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        resolved = self._model(model or "text-embedding-004")
        vectors: list[list[float]] = []
        for text in texts:
            data = await self._post(
                resolved,
                {"content": {"parts": [{"text": text}]}},
                method="embedContent",
            )
            values = (data.get("embedding") or {}).get("values") or []
            vectors.append([float(value) for value in values])
        return vectors

    async def health_check(self) -> bool:
        try:
            await self.get_models()
            return True
        except ProviderError:
            return False


__all__ = ["GeminiProvider"]
