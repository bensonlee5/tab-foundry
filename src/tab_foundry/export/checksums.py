"""Checksum helpers for export bundles."""

from __future__ import annotations

from pathlib import Path

from tab_foundry.hashing import sha256_path


def sha256_file(path: Path) -> str:
    """Return SHA256 hex digest for a file."""

    return sha256_path(path)
