from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AIProviderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str | None = Field(None, max_length=255)
    provider_type: str
    api_key: str | None = Field(None, max_length=4096)
    base_url: str | None = Field(None, max_length=1024)
    default_model: str | None = Field(None, max_length=255)
    default_inference_provider: str | None = Field(None, max_length=100)
    timeout: float | None = Field(None, gt=0, le=600)
    max_retries: int | None = Field(None, ge=0, le=10)
    config: dict[str, Any] | None = None
    is_active: bool = True
    is_default: bool = False


class AIProviderUpdate(BaseModel):
    display_name: str | None = Field(None, max_length=255)
    api_key: str | None = Field(None, max_length=4096)
    clear_api_key: bool = False
    base_url: str | None = Field(None, max_length=1024)
    default_model: str | None = Field(None, max_length=255)
    default_inference_provider: str | None = Field(None, max_length=100)
    timeout: float | None = Field(None, gt=0, le=600)
    max_retries: int | None = Field(None, ge=0, le=10)
    config: dict[str, Any] | None = None
    is_active: bool | None = None
    is_default: bool | None = None


class AIProviderResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    display_name: str | None = None
    provider_type: str
    base_url: str | None = None
    default_model: str | None = None
    default_inference_provider: str | None = None
    is_active: bool
    is_default: bool
    timeout: float | None = None
    max_retries: int | None = None
    config: dict[str, Any] | None = None
    api_key_hint: str | None = None
    health_status: str
    last_validated_at: datetime | None = None
    last_error: str | None = None
    model_count: int = 0
    created_at: datetime
    updated_at: datetime


class AIModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    model_id: str = Field(..., min_length=1, max_length=255)
    version: str | None = None
    description: str | None = None
    task: str | None = None
    max_tokens: int | None = None
    context_length: int | None = None
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    supports_json_mode: bool = False
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    is_default: bool = False


class AIModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    provider_id: str
    provider_name: str | None = None
    provider_type: str | None = None
    name: str
    model_id: str
    version: str | None = None
    description: str | None = None
    task: str | None = None
    max_tokens: int | None = None
    context_length: int | None = None
    supports_streaming: bool
    supports_tools: bool
    supports_vision: bool
    supports_embeddings: bool
    supports_audio: bool
    supports_json_mode: bool
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    is_active: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    items: list[AIModelResponse]
    total: int
    page: int
    page_size: int


class ProviderTypeResponse(BaseModel):
    type: str
    display_name: str
    requires_api_key: bool
    configured: bool
    capabilities: dict[str, Any]


class ProviderSyncResponse(BaseModel):
    provider_id: str
    discovered: int
    created: int
    updated: int
    models: list[AIModelResponse]


class ProviderHealthResponse(BaseModel):
    provider_id: str
    healthy: bool
    status: str
    error: str | None = None
    checked_at: datetime


class HuggingFaceDiscoveryResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    source: str = "huggingface"


class ProviderValidateResponse(BaseModel):
    provider_id: str
    valid: bool
    account: dict[str, Any] | None = None
    error: str | None = None
