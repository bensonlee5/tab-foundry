"""Shared helpers for corpus materialization owners."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, cast


_ACCEPTED_ONLY_MAX_GENERATED_MULTIPLIER = 4
_INITIAL_ACCEPTED_ONLY_EXPECTED_ACCEPTANCE_RATE = 0.70
_ACCEPTED_ONLY_MIN_EXPECTED_ACCEPTANCE_RATE = 0.25
_ACCEPTED_ONLY_MAX_EXPECTED_ACCEPTANCE_RATE = 0.95
_DEFAULT_MATERIALIZE_PROCESS_CAP = 8
_DEFAULT_MATERIALIZE_LARGE_MACHINE_CPU_THRESHOLD = 8
_DEFAULT_MATERIALIZE_RESERVED_CORES_LARGE_MACHINE = 2
_DEFAULT_MATERIALIZE_RESERVED_CORES_SMALL_MACHINE = 1
_SUBPROCESS_POLL_INTERVAL_SECONDS = 0.1
_STAGED_VERIFY_MODES = frozenset({"fast", "full"})


def _resolved_cpu_count(*, cpu_count: int | None = None) -> int:
    resolved = os.cpu_count() if cpu_count is None else int(cpu_count)
    if resolved is None or resolved <= 0:
        return 1
    return int(resolved)


def _materialization_reserved_cores(*, cpu_count: int | None = None) -> int:
    resolved = _resolved_cpu_count(cpu_count=cpu_count)
    if resolved >= _DEFAULT_MATERIALIZE_LARGE_MACHINE_CPU_THRESHOLD:
        return _DEFAULT_MATERIALIZE_RESERVED_CORES_LARGE_MACHINE
    return _DEFAULT_MATERIALIZE_RESERVED_CORES_SMALL_MACHINE


def _materialization_usable_cpu_budget(*, cpu_count: int | None = None) -> int:
    resolved = _resolved_cpu_count(cpu_count=cpu_count)
    reserved = _materialization_reserved_cores(cpu_count=resolved)
    return max(1, resolved - reserved)


def default_materialize_processes(*, cpu_count: int | None = None) -> int:
    usable_budget = _materialization_usable_cpu_budget(cpu_count=cpu_count)
    return min(_DEFAULT_MATERIALIZE_PROCESS_CAP, usable_budget)


def default_materialize_worker_threads(
    *,
    cpu_count: int | None = None,
    materialize_processes: int | None = None,
) -> int:
    usable_budget = _materialization_usable_cpu_budget(cpu_count=cpu_count)
    resolved_processes = _resolve_materialize_processes(materialize_processes)
    return max(1, usable_budget // max(1, resolved_processes))


def _resolve_materialize_processes(materialize_processes: int | None) -> int:
    if materialize_processes is None:
        return default_materialize_processes()
    resolved = int(materialize_processes)
    if resolved <= 0:
        raise ValueError(
            f"materialize_processes must be a positive integer, got {materialize_processes!r}"
        )
    return resolved


def _resolve_materialize_worker_threads(
    materialize_worker_threads: int | None,
    *,
    materialize_processes: int | None,
) -> int:
    if materialize_worker_threads is None:
        return default_materialize_worker_threads(
            materialize_processes=materialize_processes
        )
    resolved = int(materialize_worker_threads)
    if resolved <= 0:
        raise ValueError(
            "materialize_worker_threads must be a positive integer, "
            f"got {materialize_worker_threads!r}"
        )
    return resolved


def _git_info(root: Path) -> dict[str, Any] | None:
    if not root.exists():
        return None

    def _capture(*argv: str) -> str | None:
        completed = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    head = _capture("git", "rev-parse", "HEAD")
    if head is None:
        return None
    describe = _capture("git", "describe", "--always", "--dirty", "--tags")
    status = _capture("git", "status", "--short")
    return {
        "head": head,
        "describe": describe,
        "dirty": bool(status),
    }


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_expected_acceptance_rate(rate: float | None) -> float | None:
    if rate is None or not math.isfinite(rate):
        return None
    return min(
        _ACCEPTED_ONLY_MAX_EXPECTED_ACCEPTANCE_RATE,
        max(_ACCEPTED_ONLY_MIN_EXPECTED_ACCEPTANCE_RATE, float(rate)),
    )
