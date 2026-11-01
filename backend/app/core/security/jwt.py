from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from jose import JWTError, jwt

from app.core.config import settings


class TokenError(ValueError):
    """Raised when a JWT is invalid, expired, or of the wrong type."""


def _encode(claims: dict[str, Any]) -> str:
    return jwt.encode(  # type: ignore[no-any-return]
        claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


def create_access_token(
    subject: str,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(UTC)
    expires = expires_delta or timedelta(
        minutes=settings.jwt_access_token_expire_minutes
    )
    claims: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "nbf": now,
        "exp": now + expires,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "jti": str(uuid4()),
        "type": "access",
    }
    if extra_claims:
        claims.update(extra_claims)
    return _encode(claims)


def create_refresh_token(
    subject: str,
    jti: str | None = None,
    expires_delta: timedelta | None = None,
) -> tuple[str, str, datetime]:
    """Return ``(token, jti, expires_at)`` for a server-side tracked token."""

    now = datetime.now(UTC)
    expires = expires_delta or timedelta(days=settings.jwt_refresh_token_expire_days)
    token_id = jti or str(uuid4())
    claims: dict[str, Any] = {
        "sub": subject,
        "iat": now,
        "nbf": now,
        "exp": now + expires,
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "jti": token_id,
        "type": "refresh",
    }
    return _encode(claims), token_id, now + expires


def decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(  # type: ignore[no-any-return]
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except JWTError as exc:
        raise TokenError("Invalid or expired token") from exc


def verify_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    payload = decode_token(token)
    if payload.get("type") != expected_type:
        raise TokenError(f"Invalid token type: expected {expected_type}")
    if not payload.get("sub"):
        raise TokenError("Token is missing a subject")
    return payload


def token_fingerprint(token: str) -> str:
    """Stable SHA-256 fingerprint used to store refresh tokens server-side."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


__all__ = [
    "TokenError",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "token_fingerprint",
    "verify_token",
]
