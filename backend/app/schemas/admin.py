from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class AdminStats(BaseModel):
    users: int
    active_users: int
    conversations: int
    messages: int
    knowledge_bases: int
    documents: int
    providers: int
    models: int
    requests_24h: int
    tokens_24h: int
    cost_24h_usd: float


class AdminUserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    is_superuser: bool
    email_verified: bool
    last_login_at: datetime | None = None
    created_at: datetime


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]
    total: int
    page: int
    page_size: int


class AuditLogResponse(BaseModel):
    id: str
    user_id: str | None = None
    action: str
    resource: str
    resource_id: str | None = None
    details: dict[str, Any] | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    request_id: str | None = None
    severity: str
    success: bool
    created_at: datetime


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int
    page: int
    page_size: int


class UsageSummary(BaseModel):
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    average_latency_ms: float
    by_provider: list[dict[str, Any]]


class SystemSettingResponse(BaseModel):
    key: str
    value: Any = None
    description: str | None = None
    category: str
    is_encrypted: bool


class SystemSettingUpdate(BaseModel):
    value: Any
    description: str | None = None
    category: str = "general"
