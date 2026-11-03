"""Administrative endpoints: platform stats, users, audit logs, settings."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_superuser
from app.database.session import get_session
from app.models.chat import Conversation, Message
from app.models.knowledge import Document, KnowledgeBase
from app.models.provider import AIModel, AIProvider
from app.models.role import Role
from app.models.settings import SystemSetting
from app.models.user import User
from app.repositories.usage import AuditRepository, UsageRepository
from app.schemas.admin import (
    AdminStats,
    AdminUserListResponse,
    AdminUserResponse,
    AuditLogListResponse,
    AuditLogResponse,
    SystemSettingResponse,
    UsageSummary,
)
from app.schemas.common import MessageResponse

router = APIRouter(prefix="/admin", tags=["Administration"])


class AdminUserUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=255)
    role: str | None = Field(None, max_length=100)
    is_active: bool | None = None
    is_superuser: bool | None = None


class SystemSettingUpdateRequest(BaseModel):
    value: Any = None
    description: str | None = None
    category: str = "general"
    is_encrypted: bool = False


def _user_payload(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role_name,
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "email_verified": user.email_verified,
        "last_login_at": user.last_login_at,
        "created_at": user.created_at,
    }


def _audit_payload(log: Any) -> dict[str, Any]:
    return {
        "id": log.id,
        "user_id": log.user_id,
        "action": log.action,
        "resource": log.resource,
        "resource_id": log.resource_id,
        "details": log.details,
        "ip_address": log.ip_address,
        "user_agent": log.user_agent,
        "request_id": log.request_id,
        "severity": log.severity,
        "success": log.success,
        "created_at": log.created_at,
    }


def _setting_payload(setting: SystemSetting) -> dict[str, Any]:
    return {
        "key": setting.key,
        "value": setting.value_json if setting.value_json is not None else setting.value,
        "description": setting.description,
        "category": setting.category,
        "is_encrypted": setting.is_encrypted,
    }


async def _count(session: AsyncSession, model: Any, *where: Any) -> int:
    return int(
        await session.scalar(select(func.count()).select_from(model).where(*where))
        or 0
    )


@router.get("/stats", response_model=AdminStats)
async def platform_stats(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    since = datetime.now(UTC) - timedelta(hours=24)
    usage = await UsageRepository(session).summary(since=since)
    return {
        "users": await _count(session, User, User.is_deleted.is_(False)),
        "active_users": await _count(
            session, User, User.is_deleted.is_(False), User.is_active.is_(True)
        ),
        "conversations": await _count(
            session, Conversation, Conversation.is_deleted.is_(False)
        ),
        "messages": await _count(session, Message),
        "knowledge_bases": await _count(
            session, KnowledgeBase, KnowledgeBase.is_deleted.is_(False)
        ),
        "documents": await _count(
            session, Document, Document.is_deleted.is_(False)
        ),
        "providers": await _count(
            session, AIProvider, AIProvider.is_deleted.is_(False)
        ),
        "models": await _count(
            session, AIModel, AIModel.is_active.is_(True)
        ),
        "requests_24h": usage["requests"],
        "tokens_24h": usage["total_tokens"],
        "cost_24h_usd": usage["cost_usd"],
    }


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    search: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    where: list[Any] = [User.is_deleted.is_(False)]
    if search:
        term = f"%{search}%"
        where.append(or_(User.email.ilike(term), User.display_name.ilike(term)))
    total = await _count(session, User, *where)
    result = await session.execute(
        select(User)
        .where(*where)
        .order_by(User.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    users = result.scalars().all()
    return {
        "items": [_user_payload(user) for user in users],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.patch("/users/{user_id}", response_model=AdminUserResponse)
async def update_user(
    user_id: str,
    payload: AdminUserUpdate,
    session: AsyncSession = Depends(get_session),
    current: User = Depends(get_superuser),
) -> dict:
    user = await session.get(User, user_id)
    if user is None or user.is_deleted:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current.id and payload.is_active is False:
        raise HTTPException(status_code=400, detail="You cannot disable your own account")
    if payload.role is not None:
        role = (
            await session.execute(select(Role).where(Role.name == payload.role))
        ).scalar_one_or_none()
        if role is None:
            raise HTTPException(status_code=400, detail="Unknown role")
        user.role_id = role.id
    if payload.display_name is not None:
        user.display_name = payload.display_name
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.is_superuser is not None:
        if user.id == current.id and payload.is_superuser is False:
            raise HTTPException(
                status_code=400,
                detail="You cannot revoke your own superuser access",
            )
        user.is_superuser = payload.is_superuser
    await session.flush()
    return _user_payload(user)


@router.delete("/users/{user_id}", response_model=MessageResponse)
async def delete_user(
    user_id: str,
    session: AsyncSession = Depends(get_session),
    current: User = Depends(get_superuser),
) -> dict:
    user = await session.get(User, user_id)
    if user is None or user.is_deleted:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == current.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    user.is_deleted = True
    user.is_active = False
    user.deleted_at = datetime.now(UTC)
    await session.flush()
    return {"message": "User deleted", "status_code": 200}


@router.get("/audit-logs", response_model=AuditLogListResponse)
async def list_audit_logs(
    user_id: str | None = Query(None),
    action: str | None = Query(None),
    severity: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    logs, total = await AuditRepository(session).list_logs(
        user_id=user_id,
        action=action,
        severity=severity,
        skip=(page - 1) * page_size,
        limit=page_size,
    )
    return {
        "items": [_audit_payload(log) for log in logs],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/usage", response_model=UsageSummary)
async def usage_summary(
    hours: int = Query(24 * 7, ge=1, le=24 * 365),
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    since = datetime.now(UTC) - timedelta(hours=hours)
    repository = UsageRepository(session)
    summary = await repository.summary(since=since)
    return {**summary, "by_provider": list(await repository.by_provider(since=since))}


@router.get("/settings", response_model=list[SystemSettingResponse])
async def list_settings(
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> list[dict]:
    result = await session.execute(
        select(SystemSetting).order_by(SystemSetting.category, SystemSetting.key)
    )
    return [_setting_payload(setting) for setting in result.scalars().all()]


@router.put("/settings/{key}", response_model=SystemSettingResponse)
async def upsert_setting(
    key: str,
    payload: SystemSettingUpdateRequest,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await session.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting is None:
        setting = SystemSetting(
            key=key,
            value=None,
            value_json=payload.value if isinstance(payload.value, (dict, list)) else None,
            description=payload.description,
            category=payload.category,
            is_encrypted=payload.is_encrypted,
        )
        session.add(setting)
    else:
        if isinstance(payload.value, (dict, list)):
            setting.value_json = payload.value
            setting.value = None
        else:
            setting.value = None if payload.value is None else str(payload.value)
            setting.value_json = None
        if payload.description is not None:
            setting.description = payload.description
        setting.category = payload.category
        setting.is_encrypted = payload.is_encrypted
    await session.flush()
    return _setting_payload(setting)


@router.delete("/settings/{key}", response_model=MessageResponse)
async def delete_setting(
    key: str,
    session: AsyncSession = Depends(get_session),
    _: User = Depends(get_superuser),
) -> dict:
    result = await session.execute(
        select(SystemSetting).where(SystemSetting.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting is None:
        raise HTTPException(status_code=404, detail="Setting not found")
    await session.delete(setting)
    await session.flush()
    return {"message": "Setting deleted", "status_code": 200}
