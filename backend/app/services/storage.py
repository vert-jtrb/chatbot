"""Local upload storage with strict validation.

The storage service is deliberately small and dependency-free.  It validates
file extensions and sizes, generates collision-resistant names, and never
trusts client-provided paths.  Object storage can be added by implementing the
same two methods against S3 without touching chat or RAG code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


class StorageError(ValueError):
    """Raised when an upload is invalid or cannot be stored."""


@dataclass(slots=True)
class StoredFile:
    file_name: str
    storage_path: str
    file_size: int
    mime_type: str
    checksum: str
    data: bytes


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")
_DATA_URL = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<payload>.+)$", re.DOTALL
)


def _safe_filename(filename: str) -> str:
    name = Path(filename or "upload").name
    name = _SAFE_NAME.sub("_", name).strip(" ._") or "upload"
    return name[:180]


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def _limits() -> int:
    return max(1, settings.max_upload_size_mb) * 1024 * 1024


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    match = _DATA_URL.match(data_url or "")
    if not match:
        raise StorageError("Attachment must be a base64 encoded data URL")
    try:
        payload = base64.b64decode(match.group("payload"), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise StorageError("Attachment contains invalid base64 data") from exc
    return match.group("mime"), payload


class LocalStorage:
    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = Path(base_path or settings.local_storage_path)

    def _directory(self, kind: str) -> Path:
        directory = self.base_path / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _validate(self, filename: str, size: int, *, allowed: list[str]) -> str:
        extension = _extension(filename)
        if extension not in {item.lower() for item in allowed}:
            raise StorageError(
                f"Unsupported file type '{extension or 'unknown'}'. "
                f"Allowed: {', '.join(sorted(allowed))}"
            )
        if size <= 0:
            raise StorageError("Uploaded file is empty")
        if size > _limits():
            raise StorageError(
                f"File exceeds the maximum upload size of "
                f"{settings.max_upload_size_mb} MB"
            )
        return extension

    def save_bytes(
        self,
        *,
        kind: str,
        filename: str,
        data: bytes,
        mime_type: str,
        allowed: list[str],
    ) -> StoredFile:
        self._validate(filename, len(data), allowed=allowed)
        safe_name = _safe_filename(filename)
        self._validate(safe_name, len(data), allowed=allowed)
        destination = self._directory(kind) / f"{uuid.uuid4().hex}_{safe_name}"
        destination.write_bytes(data)
        return StoredFile(
            file_name=safe_name,
            storage_path=str(destination),
            file_size=len(data),
            mime_type=mime_type or "application/octet-stream",
            checksum=hashlib.sha256(data).hexdigest(),
            data=data,
        )

    def save_data_url(
        self,
        *,
        kind: str,
        filename: str,
        data_url: str,
        allowed: list[str],
    ) -> StoredFile:
        mime_type, data = decode_data_url(data_url)
        return self.save_bytes(
            kind=kind,
            filename=filename,
            data=data,
            mime_type=mime_type,
            allowed=allowed,
        )

    def read(self, storage_path: str) -> bytes:
        path = Path(storage_path)
        if not path.exists() or not path.is_file():
            raise StorageError("Stored file no longer exists")
        return path.read_bytes()

    def delete(self, storage_path: str) -> None:
        path = Path(storage_path)
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            # Deleting a database row must not fail because a file vanished.
            return


def attachment_kind(mime_type: str, filename: str) -> str:
    extension = _extension(filename)
    if mime_type.startswith("image/") or extension in settings.allowed_image_extensions:
        return "image"
    if mime_type.startswith("audio/") or extension in settings.allowed_audio_extensions:
        return "audio"
    return "file"


__all__ = [
    "LocalStorage",
    "StorageError",
    "StoredFile",
    "attachment_kind",
    "decode_data_url",
]
