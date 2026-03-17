"""Shared source-provenance helpers for artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import subprocess
from typing import Any, Literal, Mapping


PRODUCER_NAME = "tab-foundry"
SOURCE_PATCH_FILENAME = "source.patch"


@dataclass(slots=True, frozen=True)
class ProducerInfo:
    name: str
    version: str
    git_sha: str | None
    git_dirty: bool | None = None
    source_patch_sha256: str | None = None
    source_patch_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
        }
        if self.source_patch_sha256 is not None:
            payload["source_patch_sha256"] = self.source_patch_sha256
        if self.source_patch_path is not None:
            payload["source_patch_path"] = self.source_patch_path
        return payload

    @classmethod
    def from_mapping(cls, payload: Any, *, context: str) -> "ProducerInfo":
        if not isinstance(payload, Mapping):
            raise ValueError(f"{context} must be object")
        normalized = {str(key): value for key, value in payload.items()}
        required = ("name", "version", "git_sha")
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError(f"{context} is missing required keys: {missing}")
        git_sha = normalized["git_sha"]
        if git_sha is not None and not isinstance(git_sha, str):
            raise ValueError(f"{context}.git_sha must be string or null")
        git_dirty = normalized.get("git_dirty")
        if git_dirty is not None and not isinstance(git_dirty, bool):
            raise ValueError(f"{context}.git_dirty must be bool or null")
        source_patch_sha256 = normalized.get("source_patch_sha256")
        if source_patch_sha256 is not None and not isinstance(source_patch_sha256, str):
            raise ValueError(f"{context}.source_patch_sha256 must be string or null")
        source_patch_path = normalized.get("source_patch_path")
        if source_patch_path is not None and not isinstance(source_patch_path, str):
            raise ValueError(f"{context}.source_patch_path must be string or null")
        if (source_patch_sha256 is None) != (source_patch_path is None):
            raise ValueError(
                f"{context}.source_patch_sha256 and {context}.source_patch_path must both be set or both be null"
            )
        if source_patch_sha256 is not None and git_dirty is not True:
            raise ValueError(f"{context} patch metadata requires git_dirty=true")
        return cls(
            name=_require_non_empty_str(normalized["name"], context=f"{context}.name"),
            version=_require_non_empty_str(normalized["version"], context=f"{context}.version"),
            git_sha=git_sha,
            git_dirty=git_dirty,
            source_patch_sha256=source_patch_sha256,
            source_patch_path=source_patch_path,
        )


def source_patch_artifact_path(artifact_dir: Path) -> Path:
    return artifact_dir.expanduser().resolve() / SOURCE_PATCH_FILENAME


def resolve_current_producer(
    *,
    artifact_dir: Path | None = None,
    patch_path_mode: Literal["absolute", "relative"] = "absolute",
) -> ProducerInfo:
    version = _package_version()
    repo_root = _git_repo_root()
    if repo_root is None:
        return ProducerInfo(
            name=PRODUCER_NAME,
            version=version,
            git_sha=None,
            git_dirty=None,
        )

    git_sha, git_dirty, patch_bytes = _git_snapshot(str(repo_root))
    if git_sha is None or git_dirty is None:
        raise RuntimeError(
            "failed to resolve exact git provenance from repository checkout; "
            "cannot persist reproducible producer metadata"
        )
    if git_dirty is not True:
        return ProducerInfo(
            name=PRODUCER_NAME,
            version=version,
            git_sha=git_sha,
            git_dirty=git_dirty,
        )

    if artifact_dir is None:
        raise RuntimeError(
            "dirty git worktree requires artifact_dir so source patch provenance can be persisted"
        )
    if patch_bytes is None or not patch_bytes:
        raise RuntimeError(
            "dirty git worktree produced an empty patch; exact source provenance cannot be persisted"
        )
    patch_path = source_patch_artifact_path(artifact_dir)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch_bytes)
    recorded_patch_path = (
        patch_path.name if patch_path_mode == "relative" else str(patch_path.resolve())
    )
    return ProducerInfo(
        name=PRODUCER_NAME,
        version=version,
        git_sha=git_sha,
        git_dirty=True,
        source_patch_sha256=sha256(patch_bytes).hexdigest(),
        source_patch_path=recorded_patch_path,
    )


def _require_non_empty_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _package_version() -> str:
    try:
        return importlib.metadata.version(PRODUCER_NAME)
    except Exception:
        return "0.0.0"


def _git_repo_root() -> Path | None:
    output = _run_git_text(
        cwd=Path.cwd(),
        args=("rev-parse", "--show-toplevel"),
    )
    if output is None or not output.strip():
        return None
    return Path(output.strip()).expanduser().resolve()


@lru_cache(maxsize=4)
def _git_snapshot(repo_root_value: str) -> tuple[str | None, bool | None, bytes | None]:
    repo_root = Path(repo_root_value)
    git_sha = _run_git_text(cwd=repo_root, args=("rev-parse", "HEAD"))
    if git_sha is None or not git_sha.strip():
        return None, None, None
    status = _run_git_text(
        cwd=repo_root,
        args=("status", "--porcelain", "--untracked-files=all"),
    )
    if status is None:
        return git_sha.strip(), None, None
    if not status.strip():
        return git_sha.strip(), False, None
    return git_sha.strip(), True, _git_patch_bytes(repo_root)


def _git_patch_bytes(repo_root: Path) -> bytes:
    tracked_patch = _run_git_bytes(
        cwd=repo_root,
        args=("diff", "--binary", "HEAD", "--"),
    )
    if tracked_patch is None:
        raise RuntimeError(f"failed to capture tracked git diff from {repo_root}")
    chunks: list[bytes] = []
    if tracked_patch:
        chunks.append(tracked_patch)
        if not tracked_patch.endswith(b"\n"):
            chunks.append(b"\n")

    untracked_output = _run_git_bytes(
        cwd=repo_root,
        args=("ls-files", "--others", "--exclude-standard", "-z"),
    )
    if untracked_output is None:
        raise RuntimeError(f"failed to list untracked git files from {repo_root}")
    for raw_relpath in untracked_output.split(b"\0"):
        if not raw_relpath:
            continue
        relpath = os.fsdecode(raw_relpath)
        patch = _run_git_bytes(
            cwd=repo_root,
            args=("diff", "--no-index", "--binary", "--", "/dev/null", relpath),
            allowed_returncodes={0, 1},
        )
        if patch is None:
            raise RuntimeError(
                f"failed to capture untracked git diff for {relpath!r} from {repo_root}"
            )
        if not patch:
            continue
        chunks.append(patch)
        if not patch.endswith(b"\n"):
            chunks.append(b"\n")
    return b"".join(chunks)


def _run_git_text(
    *,
    cwd: Path,
    args: tuple[str, ...],
    allowed_returncodes: set[int] | None = None,
) -> str | None:
    result = _run_git(
        cwd=cwd,
        args=args,
        allowed_returncodes=allowed_returncodes,
    )
    if result is None:
        return None
    return result.stdout.decode("utf-8", errors="replace")


def _run_git_bytes(
    *,
    cwd: Path,
    args: tuple[str, ...],
    allowed_returncodes: set[int] | None = None,
) -> bytes | None:
    result = _run_git(
        cwd=cwd,
        args=args,
        allowed_returncodes=allowed_returncodes,
    )
    if result is None:
        return None
    return result.stdout


def _run_git(
    *,
    cwd: Path,
    args: tuple[str, ...],
    allowed_returncodes: set[int] | None = None,
) -> subprocess.CompletedProcess[bytes] | None:
    allowed = allowed_returncodes if allowed_returncodes is not None else {0}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        return None
    if result.returncode not in allowed:
        return None
    return result
