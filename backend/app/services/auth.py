from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import httpx
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security.jwt import (
    TokenError,
    create_access_token,
    create_refresh_token,
    verify_token,
)
from app.core.security.password import hash_password, verify_password
from app.models.role import Permission, Role, RolePermission
from app.models.user import AuthProvider, RefreshToken, User
from app.repositories.user import RefreshTokenRepository, UserRepository
from app.services.base import ServiceResult

DEFAULT_PERMISSIONS: dict[str, list[str]] = {
    "admin": [
        "users:read",
        "users:write",
        "providers:read",
        "providers:write",
        "knowledge:read",
        "knowledge:write",
        "chats:read",
        "chats:write",
        "admin:read",
        "admin:write",
    ],
    "user": [
        "chats:read",
        "chats:write",
        "knowledge:read",
        "knowledge:write",
        "providers:read",
    ],
}


@dataclass(slots=True)
class TokenPair:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.tokens = RefreshTokenRepository(session)

    # ------------------------------------------------------------------
    # Registration / login
    # ------------------------------------------------------------------
    async def register(
        self,
        *,
        email: str,
        password: str,
        display_name: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ServiceResult:
        existing = await self.users.get_by_email(email)
        if existing:
            return ServiceResult.error_response("Email is already registered", 409)
        await self.ensure_default_roles()
        is_first = await self.users.count() == 0
        role = await self._role_by_name("admin" if is_first else "user")
        user = await self.users.create(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            display_name=display_name.strip(),
            role_id=role.id if role else None,
            is_superuser=is_first,
            auth_provider=AuthProvider.EMAIL,
            email_verified_at=datetime.now(UTC),
        )
        tokens = await self._issue_tokens(user, ip_address=ip_address, user_agent=user_agent)
        return ServiceResult.ok(
            {"user": user, "tokens": tokens.as_dict()}, status_code=201
        )

    async def login(
        self,
        *,
        email: str,
        password: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ServiceResult:
        user = await self.users.get_by_email(email)
        if not user or not user.password_hash or not verify_password(
            password, user.password_hash
        ):
            return ServiceResult.error_response("Invalid email or password", 401)
        if not user.is_active or user.is_deleted:
            return ServiceResult.error_response("Account is disabled", 403)
        await self.users.touch_login(user.id)
        tokens = await self._issue_tokens(
            user, ip_address=ip_address, user_agent=user_agent
        )
        return ServiceResult.ok({"user": user, "tokens": tokens.as_dict()})

    # ------------------------------------------------------------------
    # Refresh rotation / revocation
    # ------------------------------------------------------------------
    async def refresh(
        self,
        refresh_token: str,
        *,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ServiceResult:
        try:
            payload = verify_token(refresh_token, expected_type="refresh")
        except TokenError as exc:
            return ServiceResult.error_response(str(exc), 401)
        from app.core.security.jwt import token_fingerprint

        stored = await self.tokens.get_by_hash(token_fingerprint(refresh_token))
        if stored is None:
            return ServiceResult.error_response("Refresh token is not recognized", 401)
        if stored.is_revoked:
            # A rotated token was replayed: revoke the whole family.
            await self.tokens.revoke_family(stored.family_id)
            return ServiceResult.error_response(
                "Refresh token reuse detected; all sessions were revoked", 401
            )
        if stored.expires_at <= datetime.now(UTC):
            await self.tokens.revoke(stored)
            return ServiceResult.error_response("Refresh token has expired", 401)
        user = await self.users.get(payload["sub"])
        if not user or not user.is_active or user.is_deleted:
            await self.tokens.revoke(stored)
            return ServiceResult.error_response("User is inactive", 401)

        token, _jti, expires_at = create_refresh_token(user.id)
        new_stored = await self.tokens.create(
            user_id=user.id,
            token_hash=token_fingerprint(token),
            family_id=stored.family_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        await self.tokens.revoke(stored, replaced_by=new_stored.token_hash)
        pair = TokenPair(
            access_token=self._access_token(user),
            refresh_token=token,
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )
        return ServiceResult.ok(pair.as_dict())

    async def logout(
        self, *, user_id: str, refresh_token: str | None = None, all_devices: bool = False
    ) -> ServiceResult:
        if all_devices:
            await self.tokens.revoke_all_for_user(user_id)
        elif refresh_token:
            from app.core.security.jwt import token_fingerprint

            stored = await self.tokens.get_by_hash(token_fingerprint(refresh_token))
            if stored and stored.user_id == user_id:
                await self.tokens.revoke_family(stored.family_id)
        return ServiceResult.ok({"message": "Logged out successfully"})

    # ------------------------------------------------------------------
    # OAuth
    # ------------------------------------------------------------------
    async def oauth_login(
        self,
        *,
        provider: str,
        code: str,
        redirect_uri: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ServiceResult:
        provider = provider.lower()
        try:
            if provider == "google":
                profile = await self._google_profile(code, redirect_uri)
            elif provider == "github":
                profile = await self._github_profile(code, redirect_uri)
            else:
                return ServiceResult.error_response(
                    f"Unsupported OAuth provider: {provider}", 400
                )
        except httpx.HTTPError as exc:
            return ServiceResult.error_response(f"OAuth exchange failed: {exc}", 502)
        if not profile.get("email"):
            return ServiceResult.error_response(
                "OAuth provider did not return a verified email", 400
            )
        await self.ensure_default_roles()
        user = await self.users.get_by_auth_provider(provider, profile["id"])
        if user is None:
            user = await self.users.get_by_email(profile["email"])
        if user is None:
            role = await self._role_by_name("user")
            user = await self.users.create(
                email=profile["email"].lower(),
                display_name=profile.get("name") or profile["email"].split("@")[0],
                avatar_url=profile.get("avatar_url"),
                auth_provider=provider,
                auth_provider_id=profile["id"],
                email_verified_at=datetime.now(UTC),
                role_id=role.id if role else None,
            )
        elif user.auth_provider == AuthProvider.EMAIL:
            user.auth_provider = AuthProvider(provider)
            user.auth_provider_id = profile["id"]
            if not user.email_verified_at:
                user.email_verified_at = datetime.now(UTC)
            await self.session.flush()
        if not user.is_active:
            return ServiceResult.error_response("Account is disabled", 403)
        tokens = await self._issue_tokens(
            user, ip_address=ip_address, user_agent=user_agent
        )
        return ServiceResult.ok({"user": user, "tokens": tokens.as_dict()})

    async def _google_profile(self, code: str, redirect_uri: str) -> dict[str, Any]:
        if not settings.oauth_google_client_id or not settings.oauth_google_client_secret:
            raise httpx.HTTPError("Google OAuth is not configured")
        async with httpx.AsyncClient(timeout=20.0) as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": settings.oauth_google_client_id,
                    "client_secret": settings.oauth_google_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]
            info = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            info.raise_for_status()
            data = info.json()
        return {
            "id": data["sub"],
            "email": data.get("email"),
            "name": data.get("name"),
            "avatar_url": data.get("picture"),
        }

    async def _github_profile(self, code: str, redirect_uri: str) -> dict[str, Any]:
        if not settings.oauth_github_client_id or not settings.oauth_github_client_secret:
            raise httpx.HTTPError("GitHub OAuth is not configured")
        async with httpx.AsyncClient(timeout=20.0) as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "code": code,
                    "client_id": settings.oauth_github_client_id,
                    "client_secret": settings.oauth_github_client_secret,
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
            token_response.raise_for_status()
            access_token = token_response.json()["access_token"]
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
            }
            info = await client.get("https://api.github.com/user", headers=headers)
            info.raise_for_status()
            data = info.json()
            email = data.get("email")
            if not email:
                emails = await client.get(
                    "https://api.github.com/user/emails", headers=headers
                )
                emails.raise_for_status()
                primary = next(
                    (
                        item
                        for item in emails.json()
                        if item.get("primary") and item.get("verified")
                    ),
                    None,
                )
                email = primary.get("email") if primary else None
        return {
            "id": str(data["id"]),
            "email": email,
            "name": data.get("name") or data.get("login"),
            "avatar_url": data.get("avatar_url"),
        }

    # ------------------------------------------------------------------
    # Role bootstrap
    # ------------------------------------------------------------------
    async def ensure_default_roles(self) -> None:
        for role_name, permissions in DEFAULT_PERMISSIONS.items():
            role = await self._role_by_name(role_name)
            if role is None:
                role = Role(
                    name=role_name,
                    description=f"Default {role_name} role",
                    is_system=True,
                )
                self.session.add(role)
                await self.session.flush()
            existing = {
                (permission.resource, permission.action)
                for permission in role.permissions
            }
            for permission_name in permissions:
                resource, action = permission_name.split(":", 1)
                if (resource, action) in existing:
                    continue
                permission = await self._permission(resource, action)
                self.session.add(
                    RolePermission(role_id=role.id, permission_id=permission.id)
                )
            await self.session.flush()

    async def _role_by_name(self, name: str) -> Role | None:
        result = await self.session.execute(select(Role).where(Role.name == name))
        return result.scalar_one_or_none()

    async def _permission(self, resource: str, action: str) -> Permission:
        result = await self.session.execute(
            select(Permission).where(
                Permission.resource == resource, Permission.action == action
            )
        )
        permission = result.scalar_one_or_none()
        if permission is None:
            permission = Permission(resource=resource, action=action)
            self.session.add(permission)
            await self.session.flush()
        return permission

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------
    async def _issue_tokens(
        self,
        user: User,
        *,
        ip_address: str | None,
        user_agent: str | None,
    ) -> TokenPair:
        family_id = str(uuid4())
        token, _jti, expires_at = create_refresh_token(user.id)
        from app.core.security.jwt import token_fingerprint

        await self.tokens.create(
            user_id=user.id,
            token_hash=token_fingerprint(token),
            family_id=family_id,
            expires_at=expires_at,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return TokenPair(
            access_token=self._access_token(user),
            refresh_token=token,
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )

    def _access_token(self, user: User) -> str:
        return create_access_token(
            user.id,
            extra_claims={
                "email": user.email,
                "role": user.role_name,
                "is_superuser": user.is_superuser,
            },
        )

    async def cleanup_expired_tokens(self) -> int:
        from sqlalchemy import delete

        result = await self.session.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < datetime.now(UTC))
        )
        return int(result.rowcount or 0)

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        raw = f"sk_{secrets.token_urlsafe(32)}"
        return raw, raw[:12], hash_password(raw)


__all__ = ["AuthService", "DEFAULT_PERMISSIONS", "TokenPair"]
