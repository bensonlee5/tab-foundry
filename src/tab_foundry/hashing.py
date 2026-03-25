"""Shared hashing helpers and digest-related constants."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


FILE_HASH_CHUNK_BYTES = 1024 * 1024
SHA256_HEX_LENGTH = 64


def sha256_path(path: Path) -> str:
    """Return the SHA256 hex digest for one file."""

    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(FILE_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str, *, encoding: str = "utf-8") -> str:
    """Return the SHA256 hex digest for one text payload."""

    return sha256(value.encode(encoding)).hexdigest()
