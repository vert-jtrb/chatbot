"""Hugging Face provider package.

Public entry points:
    HuggingFaceProvider  - chat, streaming, discovery, embeddings, ASR/TTS
    HuggingFaceClient    - low-level router/Hub client
    HuggingFaceCatalog   - model discovery with TTL cache
"""

from app.providers.ai.huggingface.client import HuggingFaceClient
from app.providers.ai.huggingface.models import HuggingFaceCatalog, HuggingFaceModel
from app.providers.ai.huggingface.provider import HuggingFaceProvider

__all__ = [
    "HuggingFaceCatalog",
    "HuggingFaceClient",
    "HuggingFaceModel",
    "HuggingFaceProvider",
]
