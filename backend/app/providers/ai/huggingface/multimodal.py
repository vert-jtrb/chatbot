"""Multimodal helpers for the Hugging Face adapter."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from huggingface_hub import InferenceClient

from app.core.config import settings
from app.providers.ai.exceptions import (
    ProviderAuthenticationError,
    ProviderUnavailableError,
)

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def image_data_url(data: bytes, filename: str = "image.png") -> str:
    mime = _IMAGE_MIME.get(Path(filename).suffix.lower())
    if not mime:
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def audio_bytes_from_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Invalid audio data URL")
    return base64.b64decode(data_url.split(",", 1)[1])


def transcribe(
    audio: bytes,
    *,
    token: str | None,
    model: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    resolved_token = token or settings.hf_token
    if not resolved_token:
        raise ProviderAuthenticationError(
            "HF_TOKEN is required for speech recognition", provider="huggingface"
        )
    kwargs = {"token": resolved_token, "timeout": settings.hf_timeout}
    if endpoint_url:
        kwargs["base_url"] = endpoint_url
    client = InferenceClient(**kwargs)
    try:
        result = client.automatic_speech_recognition(
            audio, model=model or settings.hf_asr_model
        )
    except Exception as exc:
        raise ProviderUnavailableError(
            f"Hugging Face speech recognition failed: {exc}", provider="huggingface"
        ) from exc
    if isinstance(result, str):
        return result
    return getattr(result, "text", "") or ""


def synthesize(
    text: str,
    *,
    token: str | None,
    model: str | None = None,
    endpoint_url: str | None = None,
) -> bytes:
    resolved_token = token or settings.hf_token
    if not resolved_token:
        raise ProviderAuthenticationError(
            "HF_TOKEN is required for speech synthesis", provider="huggingface"
        )
    kwargs = {"token": resolved_token, "timeout": settings.hf_timeout}
    if endpoint_url:
        kwargs["base_url"] = endpoint_url
    client = InferenceClient(**kwargs)
    try:
        return bytes(
            client.text_to_speech(text, model=model or settings.hf_tts_model)
        )
    except Exception as exc:
        raise ProviderUnavailableError(
            f"Hugging Face speech synthesis failed: {exc}", provider="huggingface"
        ) from exc


__all__ = [
    "audio_bytes_from_data_url",
    "image_data_url",
    "synthesize",
    "transcribe",
]
