"""Lightweight checkpoint path helpers shared by training-style consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def checkpoint_dir(run_dir: Path) -> Path:
    """Resolve the canonical checkpoint directory for one run directory."""

    return run_dir.expanduser().resolve() / "checkpoints"


def canonical_latest_checkpoint_path(run_dir: Path) -> Path:
    """Return the canonical latest-checkpoint path for one run directory."""

    return checkpoint_dir(run_dir) / "latest.pt"


def stage_latest_checkpoint_path(run_dir: Path, *, stage_name: str) -> Path:
    """Return the stage-scoped latest-checkpoint path for one run directory."""

    return checkpoint_dir(run_dir) / f"latest_{stage_name}.pt"


def resolve_latest_checkpoint_path(
    run_dir: Path,
    *,
    additional_run_dirs: Sequence[Path] = (),
    include_best_fallback: bool = False,
) -> Path | None:
    """Resolve the best available latest checkpoint across one or more run dirs."""

    resolved_run_dirs: list[Path] = []
    seen_run_dirs: set[Path] = set()
    for candidate in (run_dir, *additional_run_dirs):
        resolved = candidate.expanduser().resolve()
        if resolved in seen_run_dirs:
            continue
        seen_run_dirs.add(resolved)
        resolved_run_dirs.append(resolved)

    checkpoint_dirs = [checkpoint_dir(candidate) for candidate in resolved_run_dirs]
    for current_checkpoint_dir in checkpoint_dirs:
        candidate = current_checkpoint_dir / "latest.pt"
        if candidate.exists():
            return candidate.resolve()

    stage_latest_candidates: list[Path] = []
    for current_checkpoint_dir in checkpoint_dirs:
        if not current_checkpoint_dir.exists():
            continue
        for candidate in current_checkpoint_dir.glob("latest_*.pt"):
            if candidate.is_file():
                stage_latest_candidates.append(candidate)
    if stage_latest_candidates:
        return max(
            stage_latest_candidates,
            key=lambda candidate: (candidate.stat().st_mtime_ns, candidate.name),
        ).resolve()

    if include_best_fallback:
        for current_checkpoint_dir in checkpoint_dirs:
            candidate = current_checkpoint_dir / "best.pt"
            if candidate.exists():
                return candidate.resolve()
    return None
