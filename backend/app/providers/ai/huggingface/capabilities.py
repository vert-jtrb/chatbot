from __future__ import annotations

from typing import Any

from app.providers.ai.base import ModelCapabilities


def _modalities(model: dict[str, Any]) -> tuple[list[str], list[str]]:
    architecture = model.get("architecture") or {}
    inputs = architecture.get("input_modalities") or model.get("input_modalities") or []
    outputs = architecture.get("output_modalities") or model.get("output_modalities") or []
    if not inputs:
        pipeline = model.get("pipeline_tag")
        if pipeline == "text-generation":
            inputs = ["text"]
        elif pipeline in {"image-text-to-text", "visual-question-answering"}:
            inputs = ["text", "image"]
        elif pipeline in {"automatic-speech-recognition", "audio-classification"}:
            inputs = ["audio"]
        else:
            inputs = ["text"]
    if not outputs:
        outputs = ["text"]
    return list(inputs), list(outputs)


def provider_names(model: dict[str, Any]) -> list[str]:
    providers = model.get("providers")
    if isinstance(providers, list):
        return [
            item.get("provider")
            for item in providers
            if isinstance(item, dict) and item.get("provider")
        ]
    mapping = model.get("inferenceProviderMapping")
    if isinstance(mapping, dict):
        return list(mapping.keys())
    return []


def best_provider(model: dict[str, Any]) -> dict[str, Any] | None:
    providers = [
        item
        for item in (model.get("providers") or [])
        if isinstance(item, dict) and item.get("status", "live") == "live"
    ]
    if not providers:
        return None
    return max(
        providers,
        key=lambda item: (
            bool(item.get("supports_tools")),
            item.get("context_length") or 0,
            -(item.get("first_token_latency_ms") or 10**9),
        ),
    )


def detect_capabilities(model: dict[str, Any]) -> ModelCapabilities:
    inputs, _outputs = _modalities(model)
    provider = best_provider(model) or {}
    pipeline = model.get("pipeline_tag") or model.get("task") or ""
    providers = model.get("providers") or []
    supports_tools = bool(provider.get("supports_tools")) or any(
        bool(item.get("supports_tools")) for item in providers if isinstance(item, dict)
    )
    context_length = max(
        [int(item.get("context_length") or 0) for item in providers if isinstance(item, dict)]
        + [int(model.get("context_length") or 0)]
    ) or 4096
    return ModelCapabilities(
        streaming=pipeline not in {"automatic-speech-recognition", "text-to-speech"},
        tools=supports_tools,
        vision="image" in inputs
        or pipeline in {"image-text-to-text", "visual-question-answering"},
        embeddings=pipeline in {"feature-extraction", "sentence-similarity"}
        or "embedding" in str(model.get("id", "")).lower(),
        audio="audio" in inputs
        or pipeline
        in {"automatic-speech-recognition", "audio-classification", "text-to-speech"},
        json_mode=bool(
            provider.get("supports_structured_output")
            or model.get("supports_structured_output")
        ),
        max_context_length=context_length,
    )


def pricing(model: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return input/output USD per million tokens from router metadata."""

    provider = best_provider(model) or {}
    prices = provider.get("pricing") or model.get("pricing") or {}
    if not isinstance(prices, dict):
        return None, None
    return _as_float(prices.get("input")), _as_float(prices.get("output"))


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "best_provider",
    "detect_capabilities",
    "pricing",
    "provider_names",
]
