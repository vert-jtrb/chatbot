from __future__ import annotations

import secrets
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.api.middleware.rate_limit import limiter
from app.api.v1.serializers import auth_payload, user_payload
from app.core.config import settings
from app.core.security.jwt import TokenError, create_password_reset_token, verify_token
from app.core.security.password import hash_password, verify_password
from app.database.session import get_session
from app.models.user import User, UserAPIKey
from app.repositories.user import UserAPIKeyRepository, UserRepository
from app.schemas.auth import (
    APIKeyCreate,
    APIKeyCreatedResponse,
    APIKeyResponse,
    ChangePasswordRequest,
    LoginRequest,
    LoginResponse,
    OAuthAuthorizeResponse,
    OAuthLoginRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.common import MessageResponse
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = structlog.get_logger()


def _client_meta(request: Request) -> dict[str, str | None]:
    return {
        "ip_address": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }


@router.post("/register", response_model=LoginResponse, status_code=201)
@limiter.limit(f"{settings.auth_rate_limit_requests}/{settings.auth_rate_limit_period}minute")
async def register(
    request: Request,
    payload: RegisterRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = AuthService(session)
    result = await service.register(
        email=payload.email,
        password=payload.password,
        display_name=payload.display_name,
        **_client_meta(request),  # type: ignore[arg-type]
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return auth_payload(result.data["user"], result.data["tokens"])


@router.post("/login", response_model=LoginResponse)
@limiter.limit(f"{settings.auth_rate_limit_requests}/{settings.auth_rate_limit_period}minute")
async def login(
    request: Request,
    payload: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = AuthService(session)
    result = await service.login(
        email=payload.email,
        password=payload.password,
        **_client_meta(request),  # type: ignore[arg-type]
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return auth_payload(result.data["user"], result.data["tokens"])


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    payload: RefreshTokenRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await AuthService(session).refresh(
        payload.refresh_token,
        **_client_meta(request),  # type: ignore[arg-type]
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return result.data


@router.post("/logout", response_model=MessageResponse)
async def logout(
    payload: dict | None = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    payload = payload or {}
    result = await AuthService(session).logout(
        user_id=current_user.id,
        refresh_token=payload.get("refresh_token"),
        all_devices=bool(payload.get("all_devices", False)),
    )
    return {
        "message": result.data["message"],
        "status_code": status.HTTP_200_OK,
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> dict:
    return user_payload(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    payload: dict,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    allowed = {"display_name", "avatar_url", "preferences"}
    for key, value in payload.items():
        if key in allowed:
            setattr(current_user, key, value)
    await session.flush()
    return user_payload(current_user)


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if not verify_password(
        payload.current_password, current_user.password_hash or ""
    ):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.password_hash = hash_password(payload.new_password)
    await AuthService(session).tokens.revoke_all_for_user(current_user.id)
    await session.flush()
    return {"message": "Password changed successfully", "status_code": 200}


@router.post("/password-reset/request", response_model=MessageResponse)
@limiter.limit(f"{settings.auth_rate_limit_requests}/{settings.auth_rate_limit_period}minute")
async def request_password_reset(
    request: Request,
    payload: PasswordResetRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    user = await UserRepository(session).get_by_email(payload.email)
    if user is not None:
        token = create_password_reset_token(user.id)
        # Delivery is intentionally pluggable: wire an email/SMS worker here.
        # The token is only logged outside production to aid local development.
        if not settings.is_production:
            logger.info("password_reset_token_generated", email=user.email, token=token)
    return {
        "message": "If the account exists, a password reset link has been sent",
        "status_code": 200,
    }


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    payload: PasswordResetConfirm,
    session: AsyncSession = Depends(get_session),
) -> dict:
    try:
        claims = verify_token(payload.token, expected_type="password_reset")
    except TokenError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    users = UserRepository(session)
    user = await users.get(claims["sub"])
    if user is None or user.is_deleted:
        raise HTTPException(status_code=400, detail="Invalid reset token")
    user.password_hash = hash_password(payload.password)
    await AuthService(session).tokens.revoke_all_for_user(user.id)
    await session.flush()
    return {"message": "Password has been reset", "status_code": 200}


@router.get("/oauth/{provider}/authorize", response_model=OAuthAuthorizeResponse)
async def oauth_authorize(
    provider: str,
    redirect_uri: str = Query(..., min_length=1),
) -> dict:
    provider = provider.lower()
    state = secrets.token_urlsafe(24)
    if provider == "google":
        if not settings.oauth_google_client_id:
            raise HTTPException(status_code=400, detail="Google OAuth is not configured")
        from urllib.parse import urlencode

        query = urlencode(
            {
                "client_id": settings.oauth_google_client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "access_type": "offline",
                "state": state,
            }
        )
        url = f"https://accounts.google.com/o/oauth2/v2/auth?{query}"
    elif provider == "github":
        if not settings.oauth_github_client_id:
            raise HTTPException(status_code=400, detail="GitHub OAuth is not configured")
        from urllib.parse import urlencode

        query = urlencode(
            {
                "client_id": settings.oauth_github_client_id,
                "redirect_uri": redirect_uri,
                "scope": "read:user user:email",
                "state": state,
            }
        )
        url = f"https://github.com/login/oauth/authorize?{query}"
    else:
        raise HTTPException(status_code=400, detail="Unsupported OAuth provider")
    return {"authorization_url": url, "state": state}


@router.post("/oauth/{provider}/callback", response_model=LoginResponse)
async def oauth_callback(
    request: Request,
    provider: str,
    payload: OAuthLoginRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await AuthService(session).oauth_login(
        provider=provider,
        code=payload.code,
        redirect_uri=payload.redirect_uri,
        **_client_meta(request),  # type: ignore[arg-type]
    )
    if not result.success:
        raise HTTPException(status_code=result.status_code, detail=result.error)
    return auth_payload(result.data["user"], result.data["tokens"])


# ----------------------------------------------------------------------
# Personal API keys
# ----------------------------------------------------------------------
@router.get("/api-keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    keys = await UserAPIKeyRepository(session).get_active_by_user(current_user.id)
    return [
        {
            "id": key.id,
            "name": key.name,
            "key_prefix": key.key_prefix,
            "is_active": key.is_active,
            "created_at": key.created_at,
            "expires_at": key.expires_at,
            "last_used_at": key.last_used_at,
        }
        for key in keys
    ]


@router.post("/api-keys", response_model=APIKeyCreatedResponse, status_code=201)
async def create_api_key(
    payload: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    raw, prefix, hashed = AuthService.generate_api_key()
    key = UserAPIKey(
        user_id=current_user.id,
        name=payload.name,
        key_hash=hashed,
        key_prefix=prefix,
        expires_at=payload.expires_at,
    )
    session.add(key)
    await session.flush()
    return {
        "id": key.id,
        "name": key.name,
        "key_prefix": key.key_prefix,
        "is_active": key.is_active,
        "created_at": key.created_at,
        "expires_at": key.expires_at,
        "last_used_at": key.last_used_at,
        "key": raw,
    }


@router.delete("/api-keys/{key_id}", response_model=MessageResponse)
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    result = await session.execute(
        select(UserAPIKey).where(
            UserAPIKey.id == key_id,
            UserAPIKey.user_id == current_user.id,
            UserAPIKey.is_deleted.is_(False),
        )
    )
    key = result.scalar_one_or_none()
    if key is None:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    key.is_deleted = True
    from datetime import UTC, datetime

    key.deleted_at = datetime.now(UTC)
    await session.flush()
    return {"message": "API key revoked", "status_code": 200}
