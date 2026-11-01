"""Password hashing helpers.

``bcrypt`` has a hard 72-byte input limit.  Passwords are therefore pre-hashed
with SHA-256 and base64 encoded before bcrypt, which keeps arbitrary-length
Unicode passwords safe without silently truncating them.
"""

from __future__ import annotations

import base64
import hashlib

import bcrypt

_BCRYPT_ROUNDS = 12


def _prepare(password: str) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password must not be empty")
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt(_BCRYPT_ROUNDS)).decode(
        "ascii"
    )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(_prepare(plain_password), hashed_password.encode("ascii"))
    except (ValueError, TypeError):
        return False


def needs_rehash(hashed_password: str) -> bool:
    try:
        parts = hashed_password.split("$")
        return len(parts) < 4 or int(parts[2]) != _BCRYPT_ROUNDS
    except (ValueError, IndexError):
        return True


__all__ = ["hash_password", "needs_rehash", "verify_password"]
