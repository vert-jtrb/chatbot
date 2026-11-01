"""Symmetric encryption for provider credentials at rest.

The key is derived from ``ENCRYPTION_KEY`` when supplied, otherwise from the
application secret.  Only the server can decrypt values, and decrypted values
are never returned by the REST API.
"""

from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


class EncryptionError(RuntimeError):
    """Raised when a stored secret cannot be decrypted."""


def _derive_key() -> bytes:
    material = settings.encryption_key or settings.secret_key
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def generate_key() -> str:
    """Return a new Fernet key suitable for ``ENCRYPTION_KEY``."""

    return Fernet.generate_key().decode("ascii")


def encrypt_secret(value: str) -> str:
    if not value:
        raise ValueError("Cannot encrypt an empty secret")
    return Fernet(_derive_key()).encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str) -> str:
    try:
        return Fernet(_derive_key()).decrypt(value.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:  # pragma: no cover - config failure
        raise EncryptionError(
            "Unable to decrypt provider credential; check ENCRYPTION_KEY"
        ) from exc


def mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


__all__ = [
    "EncryptionError",
    "decrypt_secret",
    "encrypt_secret",
    "generate_key",
    "mask_secret",
]
